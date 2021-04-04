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
                 n_buffer: int = 2,
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
        self.init_lr = lr
        self.n_buffer = n_buffer

        self.velocity_moment_conv = np.zeros(metrics.num_conv)
        self.acceleration_moment_conv = np.zeros(metrics.num_conv)
        self.R_conv = np.zeros(metrics.num_conv)
        self.velocity_moment_fc = np.zeros(metrics.num_fc)
        self.acceleration_moment_fc = np.zeros(metrics.num_fc)
        self.R_fc = np.zeros(metrics.num_fc)

    def __setstate__(self, state):
        super(AdaS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def epoch_step(self, epoch: int) -> None:
        if epoch == 0:
            velocity_conv_rank = self.init_lr * \
                np.ones(len(self.metrics.conv_indices))
            velocity_fc_rank = self.init_lr * \
                np.ones(len(self.metrics.fc_indices))
        else:
            n_replica = self.n_buffer - min(epoch + 1, self.n_buffer)
            input_channel_replica = np.tile(
                A=self.metrics.historical_metrics[0].input_channel_S,
                reps=(n_replica, 1))
            output_channel_replica = np.tile(
                A=self.metrics.historical_metrics[0].output_channel_S,
                reps=(n_replica, 1))
            fc_channel_replica = np.tile(
                A=self.metrics.historical_metrics[0].fc_S, reps=(n_replica, 1))
            for iteration in range(self.n_buffer - n_replica):
                epoch_identifier = (epoch - self.n_buffer +
                                    n_replica + iteration + 1)
                metric = self.metrics.historical_metrics[epoch_identifier]
                input_channel_replica = np.concatenate((
                    input_channel_replica,
                    np.tile(
                        A=metric.input_channel_S,
                        reps=(1, 1))))
                output_channel_replica = np.concatenate(
                    (output_channel_replica, np.tile(
                        A=metric.output_channel_S,
                        reps=(1, 1))))
                fc_channel_replica = np.concatenate(
                    (fc_channel_replica, np.tile(
                        A=metric.fc_S,
                        reps=(1, 1))))
            x_regression = np.linspace(start=0, stop=self.n_buffer - 1,
                                       num=self.n_buffer)

            channel_replica = (input_channel_replica +
                               output_channel_replica) / 2
            # channel_replica = output_channel_replica

            """Calculate Rank Velocity"""
            velocity_conv_rank = np.polyfit(
                x=x_regression, y=channel_replica, deg=1)[0]
            velocity_fc_rank = np.polyfit(
                x=x_regression, y=fc_channel_replica, deg=1)[0][0]

        if self.step_size is not None:
            if epoch % self.step_size == 0 and epoch > 0:
                self.R_conv *= self.gamma
                self.R_fc *= self.gamma
                self.zeta *= self.gamma

        self.R_conv = self.beta * self.R_conv + self.zeta * velocity_conv_rank
        self.R_fc = self.beta * self.R_fc + self.zeta * velocity_fc_rank

        self.R_conv = np.maximum(self.R_conv, self.min_lr)
        self.R_fc = np.maximum(self.R_fc, self.min_lr)

        call_indices_conv = np.concatenate(
            (self.metrics.conv_indices, [self.metrics.fc_indices[0]]), axis=0)
        for iteration_conv in range(len(call_indices_conv) - 1):
            index_start = call_indices_conv[iteration_conv]
            index_end = call_indices_conv[iteration_conv + 1]
            self.lr_vector[index_start: index_end] = \
                self.R_conv[iteration_conv]

        call_indices_fc = np.concatenate(
            (self.metrics.fc_indices,
                [len(self.metrics.layers_info)]), axis=0)
        for iteration_fc in range(len(call_indices_fc) - 1):
            index_start = call_indices_fc[iteration_fc]
            index_end = call_indices_fc[iteration_fc + 1]
            self.lr_vector[index_start: index_end] = self.R_fc

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
