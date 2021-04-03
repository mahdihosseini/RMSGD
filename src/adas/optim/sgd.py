"""
"""
from typing import List

from torch.optim.optimizer import Optimizer, required

import numpy as np
import torch


class AdaS(Optimizer):
    """
    Vectorized SGD from torch.optim.SGD
    """

    def __init__(self,
                 params: List[torch.nn.parameter.Parameters],
                 lr: float = required,
                 beta: float = 0.8,
                 step_size: int = None,
                 gamma: float = 1,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(AdaS, self).__init__(params, defaults)

        # AdaS Specific stuff (not SGD)
        if np.less(beta, 0) or np.greater_equal(beta, 1):
            raise ValueError(f'Invalid beta: {beta}')
        if np.less(gamma, 0):
            raise ValueError(f'Invalid gamma: {gamma}')
        if np.less_equal(step_size, 0):
            raise ValueError(f'Invalid step_size: {step_size}')
        self.step_size = step_size
        self.gamma = gamma
        self.beta = beta
        self.metrics = metrics = Metrics(params=params)
        self.lr_vector = np.repeat(a=lr, repeats=len(metrics.layers))

        self.velocity_moment_conv = np.zeros(metrics.num_conv)
        self.acceleration_moment_conv = np.zeros(metrics.num_conv)
        self.R_conv = np.zeros(metrics.num_conv)
        # self.velocity_moment_fc = np.zeros(metrics.num_linear)[0]
        # self.acceleration_moment_fc = np.zeros(metrics.num_linear)[0]
        # self.R_fc = np.zeros(metrics.num_linear)[0]

    def __setstate__(self, state):
        super(AdaS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self,
             layers_todo: List[bool],
             lr_vector: List[float],
             closure: callable = None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        iteration_group = 0
        for group in self.param_groups:
            iteration_group += 1
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            iteration_p = 0
            for p in group['params']:
                if p.grad is None or not layers_todo[iteration_p]:
                    iteration_p += 1
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-lr_vector[iteration_p])
                iteration_p += 1

        return loss
