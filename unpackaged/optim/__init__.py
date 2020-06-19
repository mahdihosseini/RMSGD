"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

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
from typing import Any

import torch

from optim.adabound import AdaBound
from optim.sgd import SGDVec


def get_optimizer_scheduler(net_parameters: Any,
                            init_lr: float, optim_method: str,
                            lr_scheduler: str,
                            train_loader_len: int,
                            max_epochs: int) -> torch.nn.Module:
    optimizer = None
    scheduler = None
    if optim_method == 'SGD':
        if lr_scheduler == 'AdaS':
            optimizer = SGDVec(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=5e-4)
    elif optim_method == 'AdaM':
        optimizer = torch.optim.Adam(net_parameters, lr=init_lr)
    elif optim_method == 'AdaGrad':
        optimizer = torch.optim.Adagrad(net_parameters, lr=init_lr)
    elif optim_method == 'RMSProp':
        optimizer = torch.optim.RMSprop(net_parameters, lr=init_lr)
    elif optim_method == 'AdaDelta':
        optimizer = torch.optim.Adadelta(net_parameters, lr=init_lr)
    elif optim_method == 'AdaBound':
        optimizer = AdaBound(net_parameters, lr=init_lr)
    else:
        print(f"Adas: Warning: Unknown optimizer {optim_method}")
    if lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.5)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        first_restart_epochs = 25
        increasing_factor = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=first_restart_epochs, T_mult=increasing_factor)
    elif lr_scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=init_lr,
            steps_per_epoch=train_loader_len, epochs=max_epochs)
    elif lr_scheduler != 'AdaS':
        print(f"Adas: Warning: Unknown LR scheduler {lr_scheduler}")
    return (optimizer, scheduler)
