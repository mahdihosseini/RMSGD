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
from typing import Any, List
import sys

import torch
mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .lr_scheduler import StepLR, CosineAnnealingWarmRestarts,\
        OneCycleLR
    from .adamp import AdamPVec, AdamP
    from .sam import SAMVec, SAM
    from .sgdp import SGDPVec, SGDP
    from .novograd import NovoGrad
    from .adabound import AdaBound
    from .adashift import AdaShift
    from .adadelta import Adadelta
    from .adagrad import Adagrad
    from .rmsprop import RMSprop
    from .nosadam import NosAdam
    from .laprop import LaProp
    from .adamod import AdaMod
    from .adamax import Adamax
    from .adasls import AdaSLS
    from .nadam import NAdam
    from .padam import PAdam
    from .radam import RAdam
    # from ..AdaS import AdaS
    from .adas import Adas
    from .sgd import SGDVec
    from .adam import Adam
    from .sgd import SGD
    from .sps import SPS
    from .sls import SLS
    from .lrd import LRD
else:
    from optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts,\
        OneCycleLR
    from optim.adamp import AdamPVec, AdamP
    from optim.sam import SAMVec, SAM
    from optim.sgdp import SGDPVec, SGDP
    from optim.novograd import NovoGrad
    from optim.adabound import AdaBound
    from optim.adashift import AdaShift
    from optim.adadelta import Adadelta
    from optim.adagrad import Adagrad
    from optim.rmsprop import RMSprop
    from optim.nosadam import NosAdam
    from optim.laprop import LaProp
    from optim.adamod import AdaMod
    from optim.adamax import Adamax
    from optim.adasls import AdaSLS
    from optim.nadam import NAdam
    from optim.padam import PAdam
    from optim.radam import RAdam
    # from AdaS import AdaS
    from optim.adas import Adas
    from optim.sgd import SGDVec
    from optim.adam import Adam
    from optim.sgd import SGD
    from optim.sps import SPS
    from optim.sls import SLS
    from optim.lrd import LRD


def get_optimizer_scheduler(
        optim_method: str,
        lr_scheduler: str,
        init_lr: float,
        net_parameters: Any,
        listed_params: List[Any],
        train_loader_len: int,
        mini_batch_size: int,
        max_epochs: int,
        optimizer_kwargs=dict(),
        scheduler_kwargs=dict()) -> torch.nn.Module:
    optimizer = None
    scheduler = None
    optim_processed_kwargs = {
        k: v for k, v in optimizer_kwargs.items() if v is not None}
    scheduler_processed_kwargs = {
        k: v for k, v in scheduler_kwargs.items() if v is not None}
    if optim_method == "Adas":
        optimizer = Adas(
            params=net_parameters,
            listed_params=listed_params,
            lr=init_lr,
            **optim_processed_kwargs)
    elif optim_method == 'SGD':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " SGD optimizer in config.yaml::**kwargs")
        optimizer = SGD(
            net_parameters, lr=init_lr,
            # momentum=kwargs['momentum'],
            # weight_decay=kwargs['weight_decay']
            **optim_processed_kwargs)
    elif optim_method == 'NAG':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " NAG optimizer  in config.yaml::**kwargs")
        optimizer = SGD(
            net_parameters, lr=init_lr,
            # momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'],
            nesterov=True,
            **optim_processed_kwargs)
    elif optim_method == 'AdaM':
        optimizer = Adam(net_parameters, lr=init_lr,
                         **optim_processed_kwargs)
    elif optim_method == 'AdaGrad':
        optimizer = Adagrad(net_parameters, lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'RMSProp':
        optimizer = RMSprop(net_parameters, lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'AdaDelta':
        optimizer = Adadelta(net_parameters, lr=init_lr,
                             **optim_processed_kwargs)
    elif optim_method == 'AdaBound':
        optimizer = AdaBound(net_parameters, lr=init_lr,
                             **optim_processed_kwargs)
    elif optim_method == 'AMSBound':
        optimizer = AdaBound(net_parameters, lr=init_lr, amsbound=True,
                             **optim_processed_kwargs)
    elif optim_method == 'AdaMax':
        optimizer = Adamax(net_parameters, lr=init_lr,
                           **optim_processed_kwargs)
    elif optim_method == 'AdaMod':
        optimizer = AdaMod(net_parameters, lr=init_lr,
                           **optim_processed_kwargs)
    elif optim_method == 'AdaShift':
        optimizer = AdaShift(net_parameters, lr=init_lr,
                             **optim_processed_kwargs)
    elif optim_method == 'NAdam':
        optimizer = NAdam(net_parameters, lr=init_lr,
                          **optim_processed_kwargs)
    elif optim_method == 'NosAdam':
        optimizer = NosAdam(net_parameters, lr=init_lr,
                            **optim_processed_kwargs)
    elif optim_method == 'NovoGrad':
        optimizer = NovoGrad(net_parameters, lr=init_lr,
                             **optim_processed_kwargs)
    elif optim_method == 'PAdam':
        optimizer = PAdam(net_parameters, lr=init_lr,
                          **optim_processed_kwargs)
    elif optim_method == 'RAdam':
        optimizer = RAdam(net_parameters, lr=init_lr,
                          **optim_processed_kwargs)
    elif optim_method == 'SPS':
        optimizer = SPS(net_parameters, init_step_size=init_lr,
                        **optim_processed_kwargs)
    elif optim_method == 'SLS':
        optimizer = SLS(net_parameters, init_step_size=init_lr,
                        **optim_processed_kwargs)
    elif optim_method == 'AdaSLS':
        optimizer = AdaSLS(
            net_parameters,
            n_batches_per_epoch=train_loader_len/float(mini_batch_size),
            init_step_size=init_lr, **optim_processed_kwargs)
    elif optim_method == 'LaProp':
        optimizer = LaProp(net_parameters, lr=init_lr,
                           **optim_processed_kwargs)
    elif optim_method == 'LearningRateDropout':
        if 'lr_dropout_rate' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'lr_dropout_rate' needs to be specified for"
                "LearningRateDropout optimizer in config.yaml::**kwargs")
        optimizer = LRD(net_parameters, lr=init_lr,
                        # lr_dropout_rate=kwargs['lr_dropout_rate'],
                        **optim_processed_kwargs)
    elif optim_method == 'AdamP':
        if lr_scheduler == 'AdaS':
            optimizer = AdamPVec(
                net_parameters, lr=init_lr,
                **optim_processed_kwargs)
        else:
            optimizer = AdamP(
                net_parameters, lr=init_lr,
                **optim_processed_kwargs)
    elif optim_method == 'SAM':
        if lr_scheduler == 'AdaS':
            optimizer = SAMVec(
                net_parameters,
                SGDVec,
                lr=init_lr,
                **optim_processed_kwargs)
        else:
            optimizer = SAM(
                net_parameters, SGD, lr=init_lr,
                **optim_processed_kwargs)
    elif optim_method == 'SGDP':
        if 'momentum' not in optim_processed_kwargs.keys() or \
                'weight_decay' not in optim_processed_kwargs.keys():
            raise ValueError(
                "'momentum' and 'weight_decay' need to be specified for"
                " SGD optimizer in config.yaml::**kwargs")
        if lr_scheduler == 'AdaS':
            optimizer = SGDPVec(
                net_parameters, lr=init_lr,
                **optim_processed_kwargs)
        else:
            optimizer = SGDP(
                net_parameters, lr=init_lr,
                **optim_processed_kwargs)
    else:
        print(f"Adas: Warning: Unknown optimizer {optim_method}")
    if lr_scheduler == 'StepLR':
        if 'step_size' not in scheduler_processed_kwargs.keys() or \
                'gamma' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'step_size' and 'gamma' need to be specified for"
                "StepLR scheduler in config.yaml::**kwargs")
        scheduler = StepLR(
            optimizer,
            # step_size=kwargs['step_size'], gamma=kwargs['gamma'],
            **scheduler_processed_kwargs)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        # first_restart_epochs = 25
        # increasing_factor = 1
        if 'T_0' not in scheduler_processed_kwargs.keys() or \
                'T_mult' not in scheduler_processed_kwargs.keys():
            raise ValueError(
                "'first_restart_epochs' and 'increasing_factor' need to be "
                "specified for CosineAnnealingWarmRestarts scheduler in "
                "config.yaml::**kwargs")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_processed_kwargs['T_0'],
            T_mult=scheduler_processed_kwargs['T_mult'],
            **scheduler_processed_kwargs)
    elif lr_scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer, max_lr=init_lr,
            steps_per_epoch=train_loader_len, epochs=max_epochs,
            **scheduler_processed_kwargs)
    # elif lr_scheduler == 'AdaS':
    #     if 'beta' not in scheduler_processed_kwargs.keys() or \
    #             'p' not in scheduler_processed_kwargs.keys():
    #         raise ValueError(
    #             "'beta', 'p' need to be specified for"
    #             " AdaS scheduler in config.yaml::**kwargs")
    #     scheduler = AdaS(parameters=listed_params,
    #                      init_lr=init_lr,
    #                      # min_lr=kwargs['min_lr'],
    #                      # p=kwargs['p'],
    #                      # beta=kwargs['beta'],
    #                      # zeta=kwargs['zeta'],
    #                      **scheduler_processed_kwargs)
    elif lr_scheduler not in ['None', '']:
        print(f"Adas: Warning: Unknown LR scheduler {lr_scheduler}")
    return (optimizer, scheduler)
