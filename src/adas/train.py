"""
MIT License

Copyright (c) 2020 Mathieu Tuli

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
from argparse import Namespace as APNamespace, _SubParsersAction, \
    ArgumentParser
from typing import Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path

# import logging
import warnings
import time
import sys

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn import metrics
import pandas as pd
import numpy as np
import torch
import yaml

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, \
        OneCycleLR
    from .optim import get_optimizer_scheduler
    from .optim.sam import SAMVec, SAM
    from .early_stop import EarlyStop
    from .optim.adasls import AdaSLS
    from .models import get_network
    from .utils import parse_config
    from .metrics import Metrics
    from .models.vgg import VGG
    from .optim.sls import SLS
    from .optim.sps import SPS
    from .data import get_data
    from .optim.adas import Adas
else:
    from optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, \
        OneCycleLR
    from optim import get_optimizer_scheduler
    from optim.sam import SAMVec, SAM
    from early_stop import EarlyStop
    from optim.adasls import AdaSLS
    from models import get_network
    from utils import parse_config
    from metrics import Metrics
    from models.vgg import VGG
    from optim.sls import SLS
    from optim.sps import SPS
    from data import get_data
    from optim.adas import Adas


def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("Adas Train Args")
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
        help="Set checkpoint directory path: Default = '.adas-checkpoint'")
    sub_parser.add_argument(
        '--resume', dest='resume',
        default=None, type=str,
        help="Set checkpoint resume path: Default = None")
    # sub_parser.add_argument(
    #     '-r', '--resume', action='store_true',
    #     dest='resume',
    #     help="Flag: resume training from checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.add_argument(
        '--save-freq', default=25, type=int,
        help='Checkpoint epoch save frequency: Default = 25')
    # sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training: Default = False")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--gpu', default=0, type=int,
        help='GPU id to use: Default = 0')
    sub_parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        dest='mpd',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training: Default = False')
    sub_parser.set_defaults(mpd=False)
    sub_parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help="url used to set up distributed training:" +
             "Default = 'tcp://127.0.0.1:23456'")
    sub_parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help="distributed backend: Default = 'nccl'")
    sub_parser.add_argument(
        '--world-size', default=-1, type=int,
        help='Number of nodes for distributed training: Default = -1')
    sub_parser.add_argument(
        '--rank', default=-1, type=int,
        help='Node rank for distributed training: Default = -1')
    # sub_parser.add_argument(
    #     '--cutout', action='store_true',
    #     default=False,
    #     help='Cutout flag')
    # sub_parser.add_argument(
    #     '--n-holes', default=-1, type=int,
    #     help='N holes for cutout')
    # sub_parser.add_argument(
    #     '--cutout-length', default=-1, type=int,
    #     help='Cutout length')


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    test_loader = None
    train_sampler = None
    num_classes: int = None
    network: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler = None
    loss = None
    output_filename: Path = None
    checkpoint = None

    def __init__(
            self,
            config_path: Path,
            device: str,
            output_path: Path,
            data_path: Path,
            checkpoint_path: Path,
            resume: Path = None,
            save_freq: int = 25,
            gpu: int = None,
            ngpus_per_node: int = 0,
            world_size: int = -1,
            rank: int = -1,
            dist: bool = False,
            mpd: bool = False,
            dist_url: str = None,
            dist_backend: str = None) -> None:

        self.gpu = gpu
        self.mpd = mpd
        self.dist = dist
        self.rank = rank
        self.best_acc1 = 0.
        self.start_epoch = 0
        self.start_trial = 0
        self.device = device
        self.resume = resume
        self.dist_url = dist_url
        self.save_freq = save_freq
        self.world_size = world_size
        self.dist_backend = dist_backend
        self.ngpus_per_node = ngpus_per_node

        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.load_config(config_path, data_path)
        print("Adas: Experiment Configuration")
        print("-"*45)
        for k, v in self.config.items():
            if isinstance(v, list) or isinstance(v, dict):
                print(f"    {k:<20} {v}")
            else:
                print(f"    {k:<20} {v:<20}")
        print("-"*45)

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = parse_config(yaml.load(f))
        if self.device == 'cpu':
            warnings.warn("Using CPU will be slow")
        elif self.dist:
            if self.gpu is not None:
                config['mini_batch_size'] = int(
                    config['mini_batch_size'] / self.ngpus_per_node)
                config['num_workers'] = int(
                    (config['num_workers'] + self.ngpus_per_node - 1) /
                    self.ngpus_per_node)
        self.train_loader, self.train_sampler,\
            self.test_loader, self.num_classes = get_data(
                name=config['dataset'], root=data_path,
                mini_batch_size=config['mini_batch_size'],
                num_workers=config['num_workers'],
                cutout=config['cutout'],
                n_holes=config['n_holes'],
                length=config['cutout_length'],
                dist=self.dist)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu) if \
            config['loss'] == 'cross_entropy' else None
        if np.less(float(config['early_stop_threshold']), 0):
            print("Adas: Notice: early stop will not be used as it was " +
                  f"set to {config['early_stop_threshold']}, " +
                  "training till completion")
        elif config['optimizer'] != 'SGD' and \
                config['scheduler'] != 'Adas':
            print("Adas: Notice: early stop will not be used as it is not " +
                  "SGD with Adas, training till completion")
            config['early_stop_threshold'] = -1.
        self.early_stop = EarlyStop(
            patience=int(config['early_stop_patience']),
            threshold=float(config['early_stop_threshold']))
        cudnn.benchmark = True
        if self.resume is not None:
            if self.gpu is None:
                self.checkpoint = torch.load(str(self.resume))
            else:
                self.checkpoint = torch.load(
                    str(self.resume),
                    map_location=f'cuda:{self.gpu}')
            self.start_epoch = self.checkpoint['epoch']
            self.start_trial = self.checkpoint['trial']
            self.best_acc1 = self.checkpoint['best_acc1']
            print(f'Resuming config for trial {self.start_trial} at ' +
                  f'epoch {self.start_epoch}')
        # self.reset()

    def reset(self, learning_rate: float) -> None:
        self.performance_statistics = dict()
        self.network = get_network(name=self.config['network'],
                                   num_classes=self.num_classes)
        self.metrics = Metrics(list(self.network.parameters()),
                               p=self.config['p'])
        # TODO add other parallelisms
        if self.device == 'cpu':
            print("Resetting cpu-based network")
        elif self.dist:
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.network.cuda(self.gpu)
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network,
                    device_ids=[self.gpu])
            else:
                self.network.cuda()
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            self.network = self.network.cuda(self.gpu)
        else:
            if isinstance(self.network, VGG):
                self.network.features = torch.nn.DataParallel(
                    self.network.features)
                self.network.cuda()
            else:
                self.network = torch.nn.DataParallel(self.network)
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.config['optimizer'],
            lr_scheduler=self.config['scheduler'],
            init_lr=learning_rate,
            net_parameters=self.network.parameters(),
            listed_params=list(self.network.parameters()),
            train_loader_len=len(self.train_loader),
            mini_batch_size=self.config['mini_batch_size'],
            max_epochs=self.config['max_epochs'],
            optimizer_kwargs=self.config['optimizer_kwargs'],
            scheduler_kwargs=self.config['scheduler_kwargs'])
        self.early_stop.reset()

    def train(self) -> None:
        if not isinstance(self.config['init_lr'], list):
            list_lr = [self.config['init_lr']]
        else:
            list_lr = self.config['init_lr']
        for learning_rate in list_lr:
            lr_output_path = self.output_path / f'lr-{learning_rate}'
            lr_output_path.mkdir(exist_ok=True, parents=True)
            for trial in range(self.start_trial,
                               self.config['n_trials']):
                self.reset(learning_rate)
                if trial == self.start_trial and self.resume is not None:
                    print("Resuming Network/Optimizer")
                    self.network.load_state_dict(
                        self.checkpoint['state_dict_network'])
                    self.optimizer.load_state_dict(
                        self.checkpoint['state_dict_optimizer'])
                    self.scheduler.load_state_dict(
                        self.checkpoint['state_dict_scheduler'])
                    # else:
                    #     self.metrics.historical_metrics = \
                    #         self.checkpoint['historical_metrics']
                    epochs = range(self.start_epoch, self.config['max_epochs'])
                    self.output_filename = self.checkpoint['output_filename']
                    self.performance_statistics = self.checkpoint[
                        'performance_statistics']
                else:
                    epochs = range(0, self.config['max_epochs'])
                    self.output_filename = "results_" +\
                        f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" +\
                        f"trial={trial}_" +\
                        f"{self.config['network']}_" +\
                        f"{self.config['dataset']}_" +\
                        f"{self.config['optimizer']}" +\
                        '_'.join([f"{k}={v}" for k, v in
                                  self.config['optimizer_kwargs'].items()]) +\
                        f"_{self.config['scheduler']}" +\
                        '_'.join([f"{k}={v}" for k, v in
                                  self.config['scheduler_kwargs'].items()]) +\
                        f"_LR={learning_rate}" +\
                        ".xlsx".replace(' ', '-')
                self.output_filename = str(
                    lr_output_path / self.output_filename)
                self.run_epochs(trial, epochs)

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        for epoch in epochs:
            if self.dist:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, (train_acc1, train_acc5) = self.epoch_iteration(
                trial, epoch)
            test_loss, (test_acc1, test_acc5) = self.validate(epoch)
            end_time = time.time()
            if isinstance(self.scheduler, StepLR):
                self.scheduler.step()
            total_time = time.time()
            scheduler_string = f" w/ {self.config['scheduler']}" if \
                self.scheduler is not None else ''
            print(
                f"{self.config['optimizer']}{scheduler_string} " +
                f"on {self.config['dataset']}: " +
                f"T {trial + 1}/{self.config['n_trials']} | " +
                f"E {epoch + 1}/{epochs[-1] + 1} Ended | " +
                "E Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_acc1 * 100) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(
                    test_loss,
                    test_acc1 * 100))
            df = pd.DataFrame(data=self.performance_statistics)

            df.to_excel(self.output_filename)
            if self.early_stop(train_loss):
                print("Adas: Early stop activated.")
                break
            if not self.mpd or \
                    (self.mpd and self.rank % self.ngpus_per_node == 0):
                data = {'epoch': epoch + 1,
                        'trial': trial,
                        'config': self.config,
                        'state_dict_network': self.network.state_dict(),
                        'state_dict_optimizer': self.optimizer.state_dict(),
                        'state_dict_scheduler': self.scheduler.state_dict()
                        if self.scheduler is not None else None,
                        'best_acc1': self.best_acc1,
                        'performance_statistics': self.performance_statistics,
                        'output_filename': Path(self.output_filename).name,
                        'historical_metrics': self.metrics.historical_metrics}
                if epoch % self.save_freq == 0:
                    filename = f'trial_{trial}_epoch_{epoch}.pth.tar'
                    torch.save(data, str(self.checkpoint_path / filename))
                if np.greater(test_acc1, self.best_acc1):
                    self.best_acc1 = test_acc1
                    torch.save(
                        data, str(self.checkpoint_path / 'best.pth.tar'))
        torch.save(data, str(self.checkpoint_path / 'last.pth.tar'))

    def epoch_iteration(self, trial: int, epoch: int):
        # logging.info(f"Adas: Train: Epoch: {epoch}")
        # global net, performance_statistics, metrics, adas, config
        self.network.train()
        train_loss = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        # correct = 0
        # total = 0

        """train CNN architecture"""
        tgts = list()
        preds = list()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # start = time.time()
            # print(f'{batch_idx} / {len(train_loader)}')
            if self.gpu is not None:
                inputs = inputs.cuda(self.gpu, non_blocking=True)
            if self.device == 'cuda':
                targets = targets.cuda(self.gpu, non_blocking=True)
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, SLS) or \
                    isinstance(self.optimizer, AdaSLS):
                def closure():
                    outputs = self.network(inputs)
                    loss = self.criterion(outputs, targets)
                    return loss, outputs
                loss, outputs = self.optimizer.step(closure=closure)
            if isinstance(self.optimizer, SAM) or \
                    isinstance(self.optimizer, SAMVec):
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                outputs = self.network(inputs)
                self.criterion(outputs, targets).backward()
                if isinstance(self.scheduler, Adas):
                    self.optimizer.second_step(
                        self.metrics.layers_index_todo,
                        self.scheduler.lr_vector, zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            else:
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                # if isinstance(self.scheduler, Adas):
                #     self.optimizer.step(self.metrics.layers_index_todo,
                #                         self.scheduler.lr_vector)
                if isinstance(self.optimizer, SPS):
                    self.optimizer.step(loss=loss)
                else:
                    self.optimizer.step()

            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            if self.num_classes == 2:
                tgts.extend(targets.tolist())
                preds.extend(outputs[:, 1].tolist())
            acc1, acc5 = accuracy(
                outputs, targets, (1, min(self.num_classes, 5)),
                aoc=self.num_classes == 2)
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        self.performance_statistics[f'train_acc1_epoch_{epoch}'] = \
            top1.avg.cpu().item() / 100.
        self.performance_statistics[f'train_acc5_epoch_{epoch}'] = \
            top5.avg.cpu().item() / 100.
        self.performance_statistics[f'train_loss_epoch_{epoch}'] = \
            train_loss / (batch_idx + 1)

        io_metrics = self.metrics.evaluate(epoch)
        self.performance_statistics[f'in_S_epoch_{epoch}'] = \
            io_metrics.input_channel_S
        self.performance_statistics[f'out_S_epoch_{epoch}'] = \
            io_metrics.output_channel_S
        self.performance_statistics[f'fc_S_epoch_{epoch}'] = \
            io_metrics.fc_S
        self.performance_statistics[f'in_rank_epoch_{epoch}'] = \
            io_metrics.input_channel_rank
        self.performance_statistics[f'out_rank_epoch_{epoch}'] = \
            io_metrics.output_channel_rank
        self.performance_statistics[f'fc_rank_epoch_{epoch}'] = \
            io_metrics.fc_rank
        self.performance_statistics[f'in_condition_epoch_{epoch}'] = \
            io_metrics.input_channel_condition

        self.performance_statistics[f'out_condition_epoch_{epoch}'] = \
            io_metrics.output_channel_condition
        # for k, v in self.performance_statistics.items():
        #     try:
        #         print(k, len(v))
        #     except Exception:
        #         pass
        # if GLOBALS.ADAS is not None:
        if self.num_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(
                tgts, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print("Train AUC:", auc)
            self.performance_statistics[f'train_auc_epoch_{epoch}'] = auc
        if isinstance(self.optimizer, Adas):
            self.optimizer.epoch_step(epoch)
            kg = self.optimizer.KG
            new_kg = list()
            new_lr_vec = list()
            count = 0
            for idx, param in enumerate(self.optimizer.metrics.params):
                if len(param.shape) == 4:
                    if idx in self.optimizer.metrics.velocity_indexes:
                        new_kg.append(kg[min(len(kg)-1, count)])
                    else:
                        new_kg.append(kg[count])
                        count += 1
                    new_lr_vec.append(self.optimizer.lr_vector[idx])
            self.performance_statistics[f'rank_velocity_epoch_{epoch}'] = \
                new_kg
            self.performance_statistics[f'learning_rate_epoch_{epoch}'] = \
                new_lr_vec
        else:
            # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
            #         GLOBALS.CONFIG['optim_method'] == 'SPS':
            if isinstance(self.optimizer, SLS) or isinstance(
                    self.optimizer, SPS) or isinstance(self.optimizer, AdaSLS):
                self.performance_statistics[f'aearning_rate_epoch_{epoch}'] = \
                    self.optimizer.state['step_size']
            # elif isinstance(self.optimizer, Adas):
            #     lr_vec = self.optimizer.param_groups[0]['lr']
            #     new_lr_vec = list()
            #     count = 0
            #     for idx, param in enumerate(self.optimizer.metrics.params):
            #         if len(param.shape) == 4:
            #             if idx in self.optimizer.metrics.velocity_indexes:
            #                 new_lr_vec.append(lr_vec[count])
            #             else:
            #                 new_lr_vec.append(lr_vec[count])
            #                 count += 1
            #     print('lr', len(new_lr_vec))
            #     self.performance_statistics[
            #         f'learning_rate_epoch_{epoch}'] = \
            #         new_lr_vec

            else:
                self.performance_statistics[
                    f'learning_rate_epoch_{epoch}'] = \
                    self.optimizer.param_groups[0]['lr']
        return train_loss / (batch_idx + 1), (top1.avg.cpu().item() / 100.,
                                              top5.avg.cpu().item() / 100.)

    def validate(self, epoch: int):
        self.network.eval()
        test_loss = 0
        # correct = 0
        # total = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        tgts = list()
        preds = list()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                # inputs, targets = \
                #     inputs.to(self.device), targets.to(self.device)
                if self.gpu is not None:
                    inputs = inputs.cuda(self.gpu, non_blocking=True)
                if self.device == 'cuda':
                    targets = targets.cuda(self.gpu, non_blocking=True)
                outputs = self.network(inputs)
                if self.num_classes == 2:
                    tgts.extend(targets.tolist())
                    preds.extend(outputs[:, 1].tolist())
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                # _, predicted = outputs.max(1)
                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                acc1, acc5 = accuracy(outputs, targets, topk=(
                    1, min(self.num_classes, 5)),
                    aoc=self.num_classes == 2)
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

        if self.num_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(
                tgts, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print("Valid AUC:", auc)
            self.performance_statistics[f'test_auc_epoch_{epoch}'] = auc
        # Save checkpoint.
        # acc = 100. * correct / total
        # if acc > self.best_acc:
        #     # print('Adas: Saving checkpoint...')
        #     state = {
        #         'net': self.network.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch + 1,
        #     }
        #     if not isinstance(self.scheduler, Adas):
        #         state['historical_io_metrics'] = \
        #             self.metrics.historical_metrics
        #     torch.save(state, str(self.checkpoint_path / 'ckpt.pth'))
        #     self.best_acc = acc
        self.performance_statistics[f'test_acc1_epoch_{epoch}'] = (
            top1.avg.cpu().item() / 100.)
        self.performance_statistics[f'test_acc5_epoch_{epoch}'] = (
            top5.avg.cpu().item() / 100.)
        self.performance_statistics[f'test_loss_epoch_{epoch}'] = test_loss / (
            batch_idx + 1)
        return test_loss / (batch_idx + 1), (top1.avg.cpu().item() / 100,
                                             top5.avg.cpu().item() / 100)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets, topk=(1,), aoc: bool = False):
    if aoc and False:
        return [[0], [0]]
        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(
            targets, outputs[:, 1], pos_label=1)
        return [[metrics.auc(fpr, tpr)], [0.0]]
    else:
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.contiguous().view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path]:
    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    checkpoint_path = root_path / Path(args.checkpoint).expanduser()

    if not config_path.exists():
        raise ValueError(f"Adas: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"Adas: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"Adas: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True, parents=True)
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise ValueError("Resume path does not exist")
    return config_path, output_path, data_path, checkpoint_path,\
        Path(args.resume) if args.resume is not None else None


def main(args: APNamespace):
    print("Adas: Argument Parser Options")
    print("-"*45)
    for arg in vars(args):
        attr = getattr(args, arg)
        attr = attr if attr is not None else "None"
        print(f"    {arg:<20}: {attr:<40}")
    print("-"*45)
    args.config_path, args.output_path, \
        args.data_path, args.checkpoint_path, \
        args.resume = setup_dirs(args)
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.mpd or args.world_size > 1
    if args.mpd:
        args.world_size *= ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args: APNamespace):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    training_agent = TrainingAgent(
        config_path=args.config_path,
        device=device,
        output_path=args.output_path,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_freq=args.save_freq,
        gpu=args.gpu,
        ngpus_per_node=ngpus_per_node,
        world_size=args.world_size,
        rank=args.rank,
        dist=args.distributed,
        mpd=args.mpd,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend)
    print(f"Adas: Pytorch device is set to {training_agent.device}")
    training_agent.train()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    main(args)
