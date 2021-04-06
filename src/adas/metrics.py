"""
"""
from typing import List

import numpy as np
import torch

from .components import LayerType, IOMetrics
from .matrix_factorization import EVBMF


class Metrics:
    def __init__(self, params: List[torch.nn.parameter.Parameters]) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.layers_todo = np.ones(shape=len(params), dtype='bool')
        self.layers = list()
        self.number_of_conv = 0
        self.number_of_fc = 0
        self.historical_metrics = list()
        for iteration_block in range(len(params)):
            block_shape = params[iteration_block].shape
            if len(block_shape) == 4:
                self.layers = np.concatenate(
                    [self.layers, [LayerType.CONV]], axis=0)
                self.number_of_conv += 1
            elif len(block_shape) == 2:
                self.layers = np.concatenate(
                    [self.layers, [LayerType.FC]], axis=0)
                self.number_of_fc += 1
            else:
                self.layers = np.concatenate(
                    [self.layers, [LayerType.NON_CONV]], axis=0)
        self.final_decision_index = np.ones(
            shape=self.number_of_conv, dtype='bool')
        self.conv_indices = [i for i in range(len(self.layers)) if
                             self.layers[i] == LayerType.CONV]
        self.fc_indices = [i for i in range(len(self.layers)) if
                           self.layers[i] == LayerType.FC]
        self.non_conv_indices = [i for i in range(len(self.layers)) if
                                 self.layers[i] == LayerType.NON_CONV]

    def evaluate(self, epoch: int) -> IOMetrics:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        input_channel_rank = list()
        output_channel_rank = list()
        input_channel_S = list()
        output_channel_S = list()
        input_channel_condition = list()
        output_channel_condition = list()
        factorized_index_3 = np.zeros(len(self.conv_indices), dtype=bool)
        factorized_index_4 = np.zeros(len(self.conv_indices), dtype=bool)
        for block_index in range(len(self.conv_indices)):
            layer_tensor = self.params[self.conv_indices[block_index]].data
            tensor_size = layer_tensor.shape
            mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
            mode_3_unfold = torch.reshape(
                mode_3_unfold, [tensor_size[1], tensor_size[0] *
                                tensor_size[2] * tensor_size[3]])
            mode_4_unfold = layer_tensor
            mode_4_unfold = torch.reshape(
                mode_4_unfold, [tensor_size[0], tensor_size[1] *
                                tensor_size[2] * tensor_size[3]])
            try:
                U_approx, S_approx, V_approx = EVBMF(mode_3_unfold)
                input_channel_rank.append(
                    S_approx.shape[0] / tensor_size[1])
                low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
                low_rank_eigen = low_rank_eigen
                if len(low_rank_eigen) != 0:
                    input_channel_condition.append(
                        low_rank_eigen[0] / low_rank_eigen[-1])
                    sum_low_rank_eigen = low_rank_eigen / \
                        max(low_rank_eigen)
                    sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                else:
                    input_channel_condition.append(0)
                    sum_low_rank_eigen = 0
                factorized_index_3[block_index] = True
                input_channel_S.append(sum_low_rank_eigen / tensor_size[1])
                # NOTE never used (below)
                # mode_3_unfold_approx = torch.matmul(
                #     U_approx, torch.matmul(S_approx, torch.t(V_approx)))
            except Exception:
                U_approx = torch.zeros(mode_3_unfold.shape[0], 0)
                S_approx = torch.zeros(0, 0)
                V_approx = torch.zeros(mode_3_unfold.shape[1], 0)
                if epoch > 0:
                    # input_channel_rank.append(
                    #     variables_performance['in_rank_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # input_channel_S.append(
                    #     variables_performance['in_S_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # input_channel_condition.append(
                    #     variables_performance['in_condition_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    input_channel_rank.append(
                        self.historical_metrics[-1].input_channel_rank[block_index])
                    input_channel_S.append(
                        self.historical_metrics[-1].input_channel_S[block_index])
                    input_channel_condition.append(
                        self.historical_metrics[-1].input_channel_condition[block_index])
                else:
                    input_channel_rank.append(0)
                    input_channel_S.append(0)
                    input_channel_condition.append(0)

            try:
                U_approx, S_approx, V_approx = EVBMF(mode_4_unfold)
                output_channel_rank.append(
                    S_approx.shape[0] / tensor_size[0])
                low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
                low_rank_eigen = low_rank_eigen
                if len(low_rank_eigen) != 0:
                    output_channel_condition.append(
                        low_rank_eigen[0] / low_rank_eigen[-1])
                    sum_low_rank_eigen = low_rank_eigen / \
                        max(low_rank_eigen)
                    sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                else:
                    output_channel_condition.append(0)
                    sum_low_rank_eigen = 0
                output_channel_S.append(
                    sum_low_rank_eigen / tensor_size[0])
                # NOTE never used (below)
                factorized_index_4[block_index] = True
                # mode_4_unfold_approx = torch.matmul(
                #     U_approx, torch.matmul(S_approx, torch.t(V_approx)))
            except Exception:
                # NOTE never used (below)
                # U_approx = torch.zeros(mode_3_unfold.shape[0], 0)
                S_approx = torch.zeros(0, 0)
                # NOTE never used (below)
                # V_approx = torch.zeros(mode_3_unfold.shape[1], 0)
                if epoch > 0:
                    # output_channel_rank.append(
                    #     variables_performance['out_rank_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # output_channel_S.append(
                    #     variables_performance['out_S_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # output_channel_condition.append(
                    #     variables_performance['out_condition_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    output_channel_rank.append(
                        self.historical_metrics[-1].output_channel_rank[block_index])
                    output_channel_S.append(
                        self.historical_metrics[-1].output_channel_S[block_index])
                    output_channel_condition.append(
                        self.historical_metrics[-1].output_channel_condition[block_index])
                else:
                    output_channel_rank.append(0)
                    output_channel_S.append(0)
                    output_channel_condition.append(0)
        false_indices_3 = [i for i in range(len(factorized_index_3))
                           if factorized_index_3[i] is False]
        false_indices_4 = [i for i in range(len(factorized_index_4))
                           if factorized_index_4[i] is False]
        for false_index in false_indices_3:
            input_channel_S[false_index] = input_channel_S[false_index - 1]
            input_channel_rank[false_index] = \
                input_channel_rank[false_index - 1]
            input_channel_condition[false_index] = \
                input_channel_condition[false_index - 1]
        for false_index in false_indices_4:
            output_channel_S[false_index] = output_channel_S[false_index - 1]
            output_channel_rank[false_index] = \
                output_channel_rank[false_index - 1]
            output_channel_condition[false_index] = \
                output_channel_condition[false_index - 1]
        metrics = IOMetrics(input_channel_rank=input_channel_rank,
                            input_channel_S=input_channel_S,
                            input_channel_condition=input_channel_condition,
                            output_channel_rank=output_channel_rank,
                            output_channel_S=output_channel_S,
                            output_channel_condition=output_channel_condition,
                            fc_S=output_channel_S[-1],
                            fc_rank=output_channel_rank[-1])
        self.historical_metrics.append(metrics)
        return metrics
