"""
"""
from typing import NamedTuple, List
from enum import Enum


class LayerType(Enum):
    CONV = 1
    FC = 2
    NON_CONV = 3


class IOMetrics(NamedTuple):
    input_channel_rank: List[float]
    input_channel_S: List[float]
    input_channel_condition: List[float]
    output_channel_rank: List[float]
    output_channel_S: List[float]
    output_channel_condition: List[float]
    fc_S: float
    fc_rank: float


class LRMetrics(NamedTuple):
    rank_velocity: List[float]
    r_conv: List[float]
