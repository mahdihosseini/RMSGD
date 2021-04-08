"""
"""
from typing import List, Union

import numpy as np
import torch

from components import LayerMetrics, ConvLayerMetrics
from matrix_factorization import EVBMF


class Metrics:
    def __init__(self, params: List[torch.nn.parameter.Parameters]) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.layers_todo = np.ones(shape=len(params), dtype='bool')
        # self.layers = list()
        self.number_of_conv = 0
        self.number_of_fc = 0
        self.historical_metrics = list()
        # for iteration_block in range(len(params)):
        #     block_shape = params[iteration_block].shape
        #     if len(block_shape) == 4:
        #         self.layers = np.concatenate(
        #             [self.layers, [LayerType.CONV]], axis=0)
        #         self.number_of_conv += 1
        #     elif len(block_shape) == 2:
        #         self.layers = np.concatenate(
        #             [self.layers, [LayerType.FC]], axis=0)
        #         self.number_of_fc += 1
        #     else:
        #         self.layers = np.concatenate(
        #             [self.layers, [LayerType.NON_CONV]], axis=0)
        # self.conv_indices = [i for i in range(len(self.layers)) if
        #                      self.layers[i] == LayerType.CONV]
        # self.fc_indices = [i for i in range(len(self.layers)) if
        #                    self.layers[i] == LayerType.FC]
        # self.non_conv_indices = [i for i in range(len(self.layers)) if
        #                          self.layers[i] == LayerType.NON_CONV]

    def compute_low_rank(self, tensor: torch.Tensor,
                         epoch: int) -> torch.Tensor:
        tensor_size = tensor.shape
        # try:
        U_approx, S_approx, V_approx = EVBMF(tensor)
        rank = S_approx.shape[0] / tensor_size[1]
        low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
        low_rank_eigen = low_rank_eigen
        if len(low_rank_eigen) != 0:
            condition = low_rank_eigen[0] / low_rank_eigen[-1]
            sum_low_rank_eigen = low_rank_eigen / \
                max(low_rank_eigen)
            sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
        else:
            condition = 0
            sum_low_rank_eigen = 0
        # factorized_index_3[block_index] = True
        KG = sum_low_rank_eigen / tensor_size[1]
        # except Exception:
        #     S_approx = torch.zeros(0, 0)
        #     if epoch > 0:
        #         rank.append(
        #             self.historical_metrics[-1].input_channel_rank[block_index])
        #         KG.append(
        #             self.historical_metrics[-1].input_channel_S[block_index])
        #         condition.append(
        #             self.historical_metrics[-1].input_channel_condition[block_index])
        #     else:
        #         rank.append(0)
        #         KG.append(0)
        #         condition.append(0)
        return rank, KG, condition

    def __call__(self, epoch: int) -> List[int, Union[LayerMetrics,
                                                      ConvLayerMetrics]]:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        metrics: List[int, Union[LayerMetrics, ConvLayerMetrics]] = list()
        # factorized_index_3 = np.zeros(len(self.conv_indices), dtype=bool)
        # factorized_index_4 = np.zeros(len(self.conv_indices), dtype=bool)
        for layer_index, layer in self.params:
            if len(layer.shape) == 4:
                layer_tensor = layer.data
                tensor_size = layer_tensor.shape
                mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
                mode_3_unfold = torch.reshape(
                    mode_3_unfold, [tensor_size[1], tensor_size[0] *
                                    tensor_size[2] * tensor_size[3]])
                mode_4_unfold = layer_tensor
                mode_4_unfold = torch.reshape(
                    mode_4_unfold, [tensor_size[0], tensor_size[1] *
                                    tensor_size[2] * tensor_size[3]])
                in_rank, in_KG, in_condition = self.compute_low_rank(
                    mode_3_unfold, epoch)
                out_rank, out_KG, out_condition = self.compute_low_rank(
                    mode_4_unfold, epoch)
                metrics.append((layer_index, ConvLayerMetrics(
                    input_channel=LayerMetrics(
                        rank=in_rank,
                        KG=in_KG,
                        condition=in_condition),
                    output_channel=LayerMetrics(
                        rank=out_rank,
                        KG=out_KG,
                        condition=out_condition))))
            elif len(layer.shape) == 2:
                rank, KG, condition = self.compute_low_rank(layer, epoch)
                metrics.append((layer_index, LayerMetrics(
                    rank=rank,
                    KG=KG,
                    condition=condition)))
            else:
                metrics.append((layer_index, None))
        self.historical_metrics.append(metrics)
        return metrics
