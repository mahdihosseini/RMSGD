"""
MIT License

Copyright (c) 2019 Noah Golmant

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
@CREDIT: https://github.com/noahgolmant/pytorch-lr-dropout
"""
import torch
from torch.optim import Optimizer


# stolen from torch.optim.optimizer
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()
####################################


class LRD(Optimizer):
    r"""Implements learning rate dropout from the paper "Learning Rate Dropout"
        by Lin et. al (https://arxiv.org/abs/1912.00144)
        Original SGD implementation is taken from:
            https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        lr_dropout_rate (float, optional): Bernoulli parameter of binary mask
            for each update. Each update retained w.p. `lr_dropout_rate`
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = LRD(model.parameters(), lr=0.1, lr_dropout_rate=0.5,
                               momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr=required,
        lr_dropout_rate=0.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_dropout_rate < 0.0:
            raise ValueError(
                "Invalid learning rate dropout parameter: {}".format(
                    lr_dropout_rate)
            )
        elif lr_dropout_rate == 0.0:
            raise ValueError(
                "Learning rate dropout must be positive in order to retain some entries"
            )
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            lr_dropout_rate=lr_dropout_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(LRD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LRD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr_dropout_rate = group["lr_dropout_rate"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # construct random binary mask
                # each parameter retained with probability `lr_dropout_rate`
                device = d_p.get_device() if d_p.is_cuda else "cpu"
                mask = (
                    torch.rand_like(d_p, device=device, requires_grad=False)
                    < lr_dropout_rate
                ).type(dtype=d_p.dtype)
                # apply the mask!
                d_p.mul_(mask)

                p.data.add_(-group["lr"], d_p)

        return loss
