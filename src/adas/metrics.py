"""
"""
from typing import List, Union, Tuple

import numpy as np
import torch

from .components import LayerMetrics, ConvLayerMetrics
from .matrix_factorization import EVBMF


class Metrics:
    def __init__(self, params, linear: bool = False) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.history = list()
        mask = list()
        for param_idx, param in enumerate(params):
            if not param.requires_grad:
                mask.append(param_idx)
                continue
            param_shape = param.shape
            if not linear:
                if len(param_shape) != 4:
                    mask.append(param_idx)
            else:
                if len(param_shape) != 4 and len(param_shape) != 2:
                    mask.append(param_idx)
        self.mask = set(mask)

    def compute_low_rank(self,
                         tensor: torch.Tensor,
                         normalizer: float) -> torch.Tensor:
        if tensor.requires_grad:
            tensor = tensor.detach()
        try:
            tensor_size = tensor.shape
            if tensor_size[0] > tensor_size[1]:
                tensor = tensor.T
            U_approx, S_approx, V_approx = EVBMF(tensor)
        except RuntimeError:
            return None, None, None
        rank = S_approx.shape[0] / tensor_size[0]  # normalizer
        low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
        if len(low_rank_eigen) != 0:
            condition = low_rank_eigen[0] / low_rank_eigen[-1]
            sum_low_rank_eigen = low_rank_eigen / \
                max(low_rank_eigen)
            sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
        else:
            condition = 0
            sum_low_rank_eigen = 0
        KG = sum_low_rank_eigen / tensor_size[0]  # normalizer
        return rank, KG, condition

    def KG(self, epoch: int) -> np.ndarray:
        KG_list = list()
        for i, (index, metric) in enumerate(self.history[epoch]):
            if isinstance(metric, ConvLayerMetrics):
                KG_list.append((metric.input_channel.KG +
                                metric.output_channel.KG) / 2)
            elif isinstance(metric, LayerMetrics):
                KG_list.append(metric.KG)
        return np.array(KG_list)

    def MC(self, epoch: int) -> np.ndarray:
        MC_list = list()
        for i, (index, metric) in enumerate(self.history[epoch]):
            if isinstance(metric, ConvLayerMetrics):
                MC_list.append((metric.input_channel.condition +
                                metric.output_channel.condition) / 2)
            elif isinstance(metric, LayerMetrics):
                MC_list.append(metric.condition)
        return np.array(MC_list)

    def __call__(self) -> List[Tuple[int, Union[LayerMetrics,
                                                ConvLayerMetrics]]]:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        metrics: List[Tuple[int, Union[LayerMetrics,
                                       ConvLayerMetrics]]] = list()
        for layer_index, layer in enumerate(self.params):
            if layer_index in self.mask:
                metrics.append((layer_index, None))
                continue
            # if np.less(np.prod(layer.shape), 10_000):
            #     metrics.append((layer_index, None))
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
                    mode_3_unfold, tensor_size[1])
                if in_rank is None and in_KG is None and in_condition is None:
                    if len(self.history) > 0:
                        in_rank = self.history[-1][
                            layer_index][1].input_channel.rank
                        in_KG = self.history[-1][
                            layer_index][1].input_channel.KG
                        in_condition = self.history[-1][
                            layer_index][1].input_channel.condition
                    else:
                        in_rank = in_KG = in_condition = 0.
                out_rank, out_KG, out_condition = self.compute_low_rank(
                    mode_4_unfold, tensor_size[0])
                if out_rank is None and out_KG is None and out_condition is None:
                    if len(self.history) > 0:
                        out_rank = self.history[-1][
                            layer_index][1].output_channel.rank
                        out_KG = self.history[-1][
                            layer_index][1].output_channel.KG
                        out_condition = self.history[-1][
                            layer_index][1].output_channel.condition
                    else:
                        out_rank = out_KG = out_condition = 0.
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
                rank, KG, condition = self.compute_low_rank(
                    layer, layer.shape[0])
                if rank is None and KG is None and condition is None:
                    if len(self.history) > 0:
                        rank = self.history[-1][layer_index][1].rank
                        KG = self.history[-1][layer_index][1].KG
                        condition = self.history[-1][layer_index][1].condition
                    else:
                        rank = KG = condition = 0.
                metrics.append((layer_index, LayerMetrics(
                    rank=rank,
                    KG=KG,
                    condition=condition)))
            else:
                metrics.append((layer_index, None))
        self.history.append(metrics)
        return metrics
