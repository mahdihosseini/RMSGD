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
from typing import NamedTuple, List
from enum import Enum

# import logging


# class LogLevel(Enum):
#     '''
#     What the stdlib did not provide!
#     '''
#     DEBUG = logging.DEBUG
#     INFO = logging.INFO
#     WARNING = logging.WARNING
#     ERROR = logging.ERROR
#     CRITICAL = logging.CRITICAL
#
#     def __str__(self):
#         return self.name


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


class Statistics(NamedTuple):
    ram: float
    gpu_mem: float
    epoch_time: float
    step_time: float
