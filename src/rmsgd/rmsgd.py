"""
"""
from torch.optim.optimizer import Optimizer, required

import numpy as np
import torch

from .metrics import Metrics


class RMSGD(Optimizer):
    """
    Vectorized SGD from torch.optim.SGD
    """

    def __init__(self,
                 params,
                 lr: float = required,
                 beta: float = 0.8,
                 step_size: int = None,
                 linear: bool = False,
                 gamma: float = 1,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 params_dict: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        all_params = None
        if params_dict:
            found = False
            for i, d in enumerate(params):
                if "all_params" in d.keys():
                    all_params = d["all_params"]
                    del params[i]
                    found = True
                    break
            if not found:
                raise ValueError(
                    "If passing in a list of dictionaries for " +
                    "parameters, ensure one element in list has " +
                    "'all_params' with 'list(model.parameters())' as "
                    + "its value")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

        # Specific stuff (not SGD)
        if np.less(beta, 0) or np.greater_equal(beta, 1):
            raise ValueError(f'Invalid beta: {beta}')
        if np.less(gamma, 0):
            raise ValueError(f'Invalid gamma: {gamma}')
        if step_size is not None:
            if np.less_equal(step_size, 0):
                raise ValueError(f'Invalid step_size: {step_size}')
        self.step_size = step_size
        self.gamma = gamma
        self.beta = beta
        self.metrics = metrics = Metrics(
            params=all_params if params_dict is True else params,
            linear=linear)
        self.lr_vector = np.repeat(a=lr, repeats=len(metrics.params))
        self.velocity = np.ones(
            len(self.metrics.params) - len(self.metrics.mask)) * lr
        self.not_ready = list(range(len(self.velocity)))
        self.init_lr = lr
        self.zeta = 1.
        self.KG = 0.
        self.epoch = 0
        self.magnitude = 10 ** np.floor(np.log10(lr)) / 10.
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        KG=self.KG, step_size=step_size, gamma=gamma,
                        beta=beta, metrics=self.metrics,
                        lr_vector=self.lr_vector,
                        velocity=self.velocity, not_ready=self.not_ready,
                        init_lr=self.init_lr, zeta=self.zeta)
        super(RMSGD, self).__init__(params, defaults)

    def epoch_step(self) -> None:
        self.metrics()
        epoch = self.epoch
        self.epoch += 1
        if epoch == 0:
            self.KG = self.metrics.KG(epoch)
        else:
            KG = self.metrics.KG(epoch)
            velocity = KG - self.KG
            self.KG = KG
            self.zeta = np.nan_to_num(
                self.magnitude / (10 ** np.floor(np.log10(velocity))), nan=0)
            for idx in self.not_ready:
                if np.isclose(KG[idx], 0.):
                    velocity[idx] = (
                        self.init_lr - self.beta *
                        self.velocity[idx]) / self.zeta[idx]
                else:
                    self.not_ready.remove(idx)

        if self.step_size is not None:
            if epoch % self.step_size == 0 and epoch > 0:
                # self.lr_vector *= self.gamma
                self.velocity *= self.gamma
                velocity *= self.gamma
                self.magnitude *= 0.1
                # self.zeta *= self.gamma

        if self.init_lr < 1:
            self.zeta = 1
        if epoch > 0:
            self.velocity = np.maximum(
                self.beta * self.velocity + self.zeta * velocity, 0.)
        count = 0
        for i in range(len(self.metrics.params)):
            if i in self.metrics.mask:
                self.lr_vector[i] = self.lr_vector[i - (1 if i > 0 else 0)]
            else:
                self.lr_vector[i] = self.velocity[count]
                count += 1

    def step(self, closure: callable = None):
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

            for p_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-self.lr_vector[p_index])

        return loss
