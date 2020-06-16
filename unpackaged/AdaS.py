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
from typing import List, Any
import numpy as np

from metrics import Metrics
from components import LRMetrics  # IOMetrics


class AdaS():
    n_buffer = 2

    def __init__(self, parameters: List[Any],
                 beta: float = 0.8, zeta: float = 1.,
                 p: int = 1, init_lr: float = 3e-2,
                 min_lr: float = 1e-20) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        beta: float: AdaS gain factor [0, 1)
        eta: knowledge gain hyper-paramters [0, 1)
        init_lr: initial learning rate > 0
        min_lr: minimum possible learning rate > 0
        '''
        if beta < 0 or beta >= 1:
            raise ValueError
        # if zeta < 0 or zeta > 1:
        #     raise ValueError
        if init_lr <= 0:
            raise ValueError
        if min_lr <= 0:
            raise ValueError

        self.metrics = metrics = Metrics(parameters=parameters, p=p)
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.beta = beta
        self.zeta = zeta

        init_lr_vector = np.repeat(a=init_lr,
                                   repeats=len(metrics.layers_info))
        self.lr_vector = init_lr_vector
        self.velocity_moment_conv = np.zeros(metrics.number_of_conv)
        self.acceleration_moment_conv = np.zeros(metrics.number_of_conv)
        self.R_conv = np.zeros(metrics.number_of_conv)
        self.velocity_moment_fc = np.zeros(metrics.number_of_fc)[0]
        self.acceleration_moment_fc = np.zeros(metrics.number_of_fc)[0]
        self.R_fc = np.zeros(metrics.number_of_fc)[0]

    def step(self, epoch: int, metrics: Metrics = None) -> None:
        if epoch == 0:
            velocity_conv_rank = self.init_lr * \
                np.ones(len(self.metrics.conv_indices))
            velocity_fc_rank = self.init_lr * \
                np.ones(len(self.metrics.fc_indices))[0]
            # NOTE unused (below)
            # acceleration_conv_rank = np.zeros(len(conv_indices))
            # acceleration_fc_rank = np.zeros(len(fc_indices))[0]
            # preserving_acceleration_conv = alpha
        else:
            n_replica = AdaS.n_buffer - min(epoch + 1, AdaS.n_buffer)
            input_channel_replica = np.tile(
                A=metrics.historical_metrics[0].input_channel_S,
                reps=(n_replica, 1))
            output_channel_replica = np.tile(
                A=metrics.historical_metrics[0].output_channel_S,
                reps=(n_replica, 1))
            fc_channel_replica = np.tile(
                A=metrics.historical_metrics[0].fc_S, reps=(n_replica, 1))
            for iteration in range(AdaS.n_buffer - n_replica):
                epoch_identifier = (epoch - AdaS.n_buffer +
                                    n_replica + iteration + 1)
                metric = metrics.historical_metrics[epoch_identifier]
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
            x_regression = np.linspace(start=0, stop=AdaS.n_buffer - 1,
                                       num=AdaS.n_buffer)

            channel_replica = (input_channel_replica +
                               output_channel_replica) / 2
            # channel_replica = output_channel_replica

            """Calculate Rank Velocity"""
            velocity_conv_rank = np.polyfit(
                x=x_regression, y=channel_replica, deg=1)[0]
            velocity_fc_rank = np.polyfit(
                x=x_regression, y=fc_channel_replica, deg=1)[0][0]

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
        return LRMetrics(rank_velocity=velocity_conv_rank.tolist(),
                         r_conv=self.R_conv.tolist())
