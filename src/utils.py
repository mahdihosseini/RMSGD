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

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
from typing import Dict, Union, List

import pstats
import sys


def get_is_module():
    mod_name = vars(sys.modules[__name__])
    return True if 'adas.' in mod_name['__name__'] else False


def safe_float_cast(str_number: str) -> float:
    try:
        number = float(str_number)
    except ValueError:
        number = float('nan')
    return number


def pstats_to_dict(stats: pstats.Stats) -> List[Dict[str, Union[str, float]]]:
    formatted_stats = list()
    stats = 'ncalls'+stats.split('ncalls')[-1]
    stats = [line.rstrip().split(None, 5) for line in
             stats.split('\n')]
    for stat in stats[1:]:
        stats_dict = dict()
        if len(stat) >= 5:
            stats_dict['n_calls'] = stat[0]
            stats_dict['tot_time'] = stat[1]
            stats_dict['per_call1'] = stat[2]
            stats_dict['cum_time'] = stat[3]
            stats_dict['per_call2'] = stat[4]
            name = stat[5].split(':')
            stats_dict['name'] = \
                f"{name[0].split('/')[-1]}_line(function)_{name[1]}"
            formatted_stats.append(stats_dict)
    return formatted_stats


def smart_string_to_float(
    string: str,
        e: str = 'could not convert string to float') -> float:
    try:
        ret = float(string)
        return ret
    except ValueError:
        raise ValueError(e)


def smart_string_to_int(
    string: str,
        e: str = 'could not convert string to int') -> int:
    try:
        ret = int(string)
        return ret
    except ValueError:
        raise ValueError(e)
    return float('inf')


def parse_config(
    config: Dict[str, Union[str, float, int]]) -> Dict[
        str, Union[str, float, int]]:
    valid_dataset = ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet', 'MHIST']
    if config['dataset'] not in valid_dataset:
        raise ValueError(
            f"config.yaml: unknown dataset {config['dataset']}. " +
            f"Must be one of {valid_dataset}")
    valid_models = {
        'AlexNet', ', DenseNet201', 'DenseNet169', 'DenseNet161',
        'DenseNet121', 'GoogLeNet', 'InceptionV3', 'MNASNet_0_5',
        'MNASNet_0_75', 'MNASNet_1', 'MNASNet_1_3', 'MobileNetV2',
        'MobileNetV2CIFAR', 'SENet18CIFAR', 'ShuffleNetV2CIFAR',
        'ResNet18', 'ResNet34', 'ResNet34CIFAR', 'ResNet50', 'ResNet50CIFAR',
        'ResNet101', 'ResNet101CIFAR', 'ResNet152',
        'ResNext50', 'ResNext101', 'ResNeXtCIFAR', 'WideResNet50',
        'WideResNet101',
        'ShuffleNetV2_0_5', 'ShuffleNetV2_1', 'ShuffleNetV2_1_5',
        'ShuffleNetV2_2', 'SqueezeNet_1', 'SqueezeNet_1_1', 'VGG11',
        'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19',
        'VGG19_BN', 'EfficientNetB4', 'EfficientNetB0CIFAR', 'VGG16CIFAR',
        'DenseNet121CIFAR', 'ResNet18CIFAR',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7', 'EfficientNetB8',
        'EfficientNetB0CIFAR', 'EfficientNetB1CIFAR', 'EfficientNetB2CIFAR',
        'EfficientNetB3CIFAR', 'EfficientNetB4CIFAR', 'EfficientNetB5CIFAR',
        'EfficientNetB6CIFAR', 'EfficientNetB7CIFAR', 'EfficientNetB8CIFAR',
    }
    if config['network'] not in valid_models:
        raise ValueError(
            f"config.yaml: unknown model {config['network']}." +
            f"Must be one of {valid_models}")

    if config['scheduler'] == 'AdaS' and config['optimizer'] not in ['SGD', 'AdamP', 'SGDP', 'SAM']:
        raise ValueError(
            "config.yaml: AdaS can't be used with this optimizer.")

    config['n_trials'] = smart_string_to_int(
        config['n_trials'],
        e='config.yaml: n_trials must be an int')
    # config['beta'] = smart_string_to_float(
    #     config['beta'],
    #     e='config.yaml: beta must be a float')
    e = 'config.yaml: init_lr must be a float or list of floats'
    if not isinstance(config['init_lr'], str):
        if isinstance(config['init_lr'], list):
            for i, lr in enumerate(config['init_lr']):
                if config['init_lr'][i] != 'auto':
                    config['init_lr'][i] = smart_string_to_float(lr, e=e)
        else:
            config['init_lr'] = smart_string_to_float(config['init_lr'], e=e)
    else:
        if config['init_lr'] != 'auto':
            raise ValueError(e)
    config['max_epochs'] = smart_string_to_int(
        config['max_epochs'],
        e='config.yaml: max_epochs must be an int')
    config['early_stop_threshold'] = smart_string_to_float(
        config['early_stop_threshold'],
        e='config.yaml: early_stop_threshold must be a float')
    config['early_stop_patience'] = smart_string_to_int(
        config['early_stop_patience'],
        e='config.yaml: early_stop_patience must be an int')
    config['mini_batch_size'] = smart_string_to_int(
        config['mini_batch_size'],
        e='config.yaml: mini_batch_size must be an int')
    # config['min_lr'] = smart_string_to_float(
    #     config['min_lr'],
    #     e='config.yaml: min_lr must be a float')
    # config['zeta'] = smart_string_to_float(
    #     config['zeta'],
    #     e='config.yaml: zeta must be a float')
    config['p'] = smart_string_to_int(
        config['p'],
        e='config.yaml: p must be an int')
    config['num_workers'] = smart_string_to_int(
        config['num_workers'],
        e='config.yaml: num_works must be an int')
    if config['loss'] != 'cross_entropy':
        raise ValueError('config.yaml: loss must be cross_entropy')
    for k, v in config['optimizer_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['optimizer_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['optimizer_kwargs'][k] = smart_string_to_float(v)
    for k, v in config['scheduler_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['scheduler_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['scheduler_kwargs'][k] = smart_string_to_float(v)
    if config['cutout']:
        if config['n_holes'] < 0 or config['cutout_length'] < 0:
            raise ValueError('N holes and length for cutout not set')
    return config
