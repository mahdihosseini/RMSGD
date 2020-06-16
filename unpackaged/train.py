"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from argparse import Namespace as APNamespace, \
    _SubParsersAction, ArgumentParser
from typing import Tuple
from pathlib import Path

# import logging
import time

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml

# from .test import main as test_main
# from .utils import progress_bar
from early_stop import EarlyStop
from optim import get_optimizer_scheduler
from metrics import Metrics
from models import get_net
from data import get_data
from AdaS import AdaS


net = None
performance_statistics = None
criterion = None
best_acc = 0
metrics = None
adas = None
checkpoint_path = None
early_stop = None
config = None


def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS Train Args")
    # print("---------------------------------\n")
    # sub_parser.add_argument(
    #     '-vv', '--very-verbose', action='store_true',
    #     dest='very_verbose',
    #     help="Set flask debug mode")
    # sub_parser.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set flask debug mode")
    # sub_parser.set_defaults(verbose=False)
    # sub_parser.set_defaults(very_verbose=False)
    # sub_parser.add_argument(
    #     '--beta', dest='beta',
    #     default=0.8, type=float,
    #     help="set beta hyper-parameter")
    # sub_parser.add_argument(
    #     '--zeta', dest='zeta',
    #     default=1.0, type=float,
    #     help="set zeta hyper-parameter")
    # sub_parser.add_argument(
    #     '-p', dest='p',
    #     default=2, type=int,
    #     help="set power (p) hyper-parameter")
    # sub_parser.add_argument(
    #     '--init-lr', dest='init_lr',
    #     default=3e-2, type=float,
    #     help="set initial learning rate")
    # sub_parser.add_argument(
    #     '--min-lr', dest='min_lr',
    #     default=3e-2, type=float,
    #     help="set minimum learning rate")
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='.adas-data', type=str,
        help="Set data directory path: Default = '.adas-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='.adas-output', type=str,
        help="Set output directory path: Default = '.adas-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.adas-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.adas-checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.add_argument(
        '-r', '--resume', action='store_true',
        dest='resume',
        help="Flag: resume training from checkpoint")
    sub_parser.set_defaults(verbose=False)


def get_loss(loss: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else \
        None


def main(args: APNamespace):
    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    global checkpoint_path, config
    checkpoint_path = root_path / Path(args.checkpoint).expanduser()

    if not config_path.exists():
        # logging.critical(f"AdaS: Config path {config_path} does not exist")
        print(f"AdaS: Config path {config_path} does not exist")
        raise ValueError
    if not data_path.exists():
        print(f"AdaS: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AdaS: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not checkpoint_path.exists():
        if args.resume:
            print(f"AdaS: Cannot resume from checkpoint without specifying " +
                  "checkpoint dir")
            raise ValueError
        checkpoint_path.mkdir(exist_ok=True, parents=True)

    with config_path.open() as f:
        config = yaml.load(f)
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    print(f"    {'checkpoint':<20}: " +
          f"{str(Path(args.root) / args.checkpoint):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    print(f"    {'resume':<20}: {'True' if args.resume else 'False':<20}")
    print("\nAdas: Train: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    for k, v in config.items():
        print(f"    {k:<20} {v:<20}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"AdaS: Pytorch device is set to {device}")
    global best_acc
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if np.less(float(config['early_stop_threshold']), 0):
        print("AdaS: Notice: early stop will not be used as it was set to " +
              "{early_stop}, training till completion.")

    for trial in range(config['n_trials']):
        device
        # Data
        # logging.info("Adas: Preparing Data")
        train_loader, test_loader = get_data(
            root=data_path,
            dataset=config['dataset'],
            mini_batch_size=config['mini_batch_size'])
        global performance_statistics, net, metrics, adas
        performance_statistics = {}

        # logging.info("AdaS: Building Model")
        net = get_net(config['network'], num_classes=10 if config['dataset'] ==
                      'CIFAR10' else 100 if config['dataset'] == 'CIFAR100'
                      else 1000 if config['dataset'] == 'ImageNet' else 10)
        metrics = Metrics(list(net.parameters()),
                          p=config['p'])
        if config['lr_scheduler'] == 'AdaS':
            adas = AdaS(parameters=list(net.parameters()),
                        beta=config['beta'],
                        zeta=config['zeta'],
                        init_lr=float(config['init_lr']),
                        min_lr=float(config['min_lr']),
                        p=config['p'])

        net = net.to(device)

        global criterion
        criterion = get_loss(config['loss'])

        optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=net.parameters(),
            init_lr=float(config['init_lr']),
            optim_method=config['optim_method'],
            lr_scheduler=config['lr_scheduler'],
            train_loader_len=len(train_loader),
            max_epochs=int(config['max_epoch']))
        early_stop = EarlyStop(patience=int(config['early_stop_patience']),
                               threshold=float(config['early_stop_threshold']))

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print("Adas: Resuming from checkpoint...")
            checkpoint = torch.load(str(checkpoint_path / 'ckpt.pth'))
            # if checkpoint_path.is_dir():
            #     checkpoint = torch.load(str(checkpoint_path / 'ckpt.pth'))
            # else:
            #     checkpoint = torch.load(str(checkpoint_path))
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            if adas is not None:
                metrics.historical_metrics = \
                    checkpoint['historical_io_metrics']

        # model_parameters = filter(lambda p: p.requires_grad,
        #                           net.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(params)
        epochs = range(start_epoch, start_epoch + config['max_epoch'])
        for epoch in epochs:
            start_time = time.time()
            # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")
            train_loss, train_accuracy = epoch_iteration(
                train_loader, epoch, device, optimizer, scheduler)
            end_time = time.time()
            if config['lr_scheduler'] == 'StepLR':
                scheduler.step()
            test_loss, test_accuracy = test_main(test_loader, epoch, device)
            total_time = time.time()
            print(
                f"AdaS: Trial {trial}/{config['n_trials'] - 1} | " +
                f"Epoch {epoch}/{epochs[-1]} Ended | " +
                "Total Time: {:.3f}s | ".format(total_time - start_time) +
                "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_accuracy) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(test_loss,
                                                                test_accuracy))
            df = pd.DataFrame(data=performance_statistics)
            if config['lr_scheduler'] == 'AdaS':
                xlsx_name = \
                    f"config['optim_method']_AdaS_trial={trial}_" +\
                    f"beta={config['beta']}_initlr={config['init_lr']}_" +\
                    f"net={config['network']}_dataset={config['dataset']}.xlsx"
            else:
                xlsx_name = \
                    f"config['optim_method']_config['lr_scheduler']_" +\
                    f"trial={trial}_initlr={config['init_lr']}" +\
                    f"net={config['network']}_dataset={config['dataset']}.xlsx"

            df.to_excel(str(output_path / xlsx_name))
            if early_stop(train_loss):
                print("AdaS: Early stop activated.")
                break


def test_main(test_loader, epoch: int, device) -> Tuple[float, float]:
    global best_acc, performance_statistics, net, criterion, checkpoint_path
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(
            #     batch_idx, len(test_loader),
            #     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss / (batch_idx + 1), 100. * correct / total,
            #        correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        # print('Adas: Saving checkpoint...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
        }
        if adas is not None:
            state['historical_io_metrics'] = metrics.historical_metrics
        torch.save(state, str(checkpoint_path / 'ckpt.pth'))
        # if checkpoint_path.is_dir():
        #     torch.save(state, str(checkpoint_path / 'ckpt.pth'))
        # else:
        #     torch.save(state, str(checkpoint_path))
        best_acc = acc
    performance_statistics['acc_epoch_' + str(epoch)] = acc / 100
    return test_loss / (batch_idx + 1), acc


def epoch_iteration(train_loader, epoch: int,
                    device, optimizer, scheduler) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    global net, performance_statistics, metrics, adas
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if config['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if adas is not None:
            optimizer.step(metrics.layers_index_todo,
                           adas.lr_vector)
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if config['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))
    performance_statistics['Train_loss_epoch_' +
                           str(epoch)] = train_loss / (batch_idx + 1)

    io_metrics = metrics.evaluate(epoch)
    performance_statistics['in_S_epoch_' +
                           str(epoch)] = io_metrics.input_channel_S
    performance_statistics['out_S_epoch_' +
                           str(epoch)] = io_metrics.output_channel_S
    performance_statistics['fc_S_epoch_' +
                           str(epoch)] = io_metrics.fc_S
    performance_statistics['in_rank_epoch_' +
                           str(epoch)] = io_metrics.input_channel_rank
    performance_statistics['out_rank_epoch_' +
                           str(epoch)] = io_metrics.output_channel_rank
    performance_statistics['fc_rank_epoch_' +
                           str(epoch)] = io_metrics.fc_rank
    performance_statistics['in_condition_epoch_' +
                           str(epoch)] = io_metrics.input_channel_condition
    performance_statistics['out_condition_epoch_' +
                           str(epoch)] = io_metrics.output_channel_condition
    if adas is not None:
        lrmetrics = adas.step(epoch, metrics)
        performance_statistics['rank_velocity_epoch_' +
                               str(epoch)] = lrmetrics.rank_velocity
        performance_statistics['learning_rate_' +
                               str(epoch)] = lrmetrics.r_conv
    return train_loss / (batch_idx + 1), 100. * correct / total


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    main(args)
