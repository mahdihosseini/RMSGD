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
from typing import Union, List
from datetime import datetime
import sys

# import logging

import numpy as np

mod_name = vars(sys.modules[__name__])['__package__']

if mod_name:
    from .metrics import Metrics
else:
    from metrics import Metrics


def compute_rank(metrics: Metrics) -> float:
    per_S_zero = np.mean([np.mean(np.isclose(
        metric.input_channel_S + metric.output_channel_S,
        0).astype(np.float16)) for
        metric in metrics.historical_metrics])
    return per_S_zero


def compute_rate_of_change(rank_history: List[float]) -> float:
    if len(rank_history) == 1:
        return [1.]
    cumprod = np.cumprod(rank_history)
    output = list()
    for i in range(len(rank_history)):
        temp = cumprod[:i+1] - (rank_history[i] * (i+1)) + 1e-7
        if i == 0:
            output.append(1.0)
        else:
            temp = (temp[i] - temp[i-1])/temp[i]
            output.append(temp)
    return output


def auto_lr(training_agent,
            min_lr: float = 1e-4,
            max_lr: float = 0.1,
            num_split: int = 20,
            min_delta: float = 5e-3,
            lr_delta: float = 3e-5,
            epochs: Union[range, List[int]] = range(0, 5),
            exit_counter_thresh: int = 6,
            power: float = 0.8):
    # ###### LR RANGE STUFF #######
    learning_rates = np.geomspace(min_lr, max_lr, num_split)
    lr_idx = 0
    exit_counter = 0
    first_run = True
    output_history = list()
    rank_history = list()
    cur_rank = -1
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    while True:
        with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
            f.write('lr,rank,msg\n')
            for (lr, rank, msg) in output_history:
                f.write(f'{lr},{rank},{msg}\n')
        if lr_idx == len(learning_rates):
            min_lr = learning_rates[-2]
            max_lr = float(learning_rates[-1]) * 1.5
            print("LR Range Test: Reached End of Grid: Expanding.")
            learning_rates = np.geomspace(min_lr, max_lr, num_split)
            # rank_history = list()
            output_history.append(
                (learning_rates[lr_idx - 1], -1, 'end-reached'))
            lr_idx = 0
            cur_rank = -1
            continue
        if np.less(np.abs(np.subtract(min_lr, max_lr)), lr_delta):
            print(
                "LR Range Test Complete: LR Delta: Final LR Range is "
                f"{min_lr}-{max_lr}")
            output_history.append(
                (learning_rates[lr_idx], cur_rank, 'exit-delta'))
            break
        training_agent.reset(learning_rates[lr_idx])
        training_agent.output_filename = training_agent.output_path / 'auto-lr'
        training_agent.output_filename.mkdir(exist_ok=True, parents=True)
        string_name = \
            "auto_lr_results_" +\
            f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" +\
            f"trial=AdaS_trial={0}_" +\
            f"network={training_agent.config['network']}_" +\
            f"dataset={training_agent.config['dataset']}" +\
            f"optimizer={training_agent.config['optimizer']}" +\
            '_'.join([f"{k}={v}" for k, v in
                      training_agent.config['optimizer_kwargs'].items()]) +\
            f"_scheduler={training_agent.config['scheduler']}" +\
            '_'.join([f"{k}={v}" for k, v in
                      training_agent.config['scheduler_kwargs'].items()]) +\
            f"_learning_rate={learning_rates[lr_idx]}" +\
            ".csv".replace(' ', '-')
        training_agent.output_filename = str(
            training_agent.output_filename / string_name)
        training_agent.run_epochs(trial=0, epochs=epochs)

        # ###### LR RANGE STUFF #######
        cur_rank = compute_rank(training_agent.metrics)
        rank_history.append(cur_rank)
        rate_of_change = np.cumprod(rank_history) ** power
        print(f"LR Range Test: Cur. Grid Space: {learning_rates}")
        print(f"LR Range Test: Cur. LR: {learning_rates[lr_idx]}")
        print(f"LR Range Test: Cur. Rank: {cur_rank}")
        if np.isclose(cur_rank, 1.):
            output_history.append((learning_rates[lr_idx],
                                   cur_rank, 'all-zero'))
            min_lr *= 5
            if np.greater(min_lr, max_lr):
                max_lr = min_lr*1.5
            learning_rates = np.geomspace(min_lr, max_lr, num_split)
            rank_history = list()
            print("LR Range Test: All zero, new range: " +
                  f"{learning_rates}")
            lr_idx = 0
            cur_rank = -1
            continue
        if lr_idx == 0:
            if first_run and np.less(cur_rank, 0.5):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'early-cross'))
                min_lr /= 10
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Crossed thresh early, new range: " +
                      f"{learning_rates}")
                cur_rank = -1
                continue
        else:
            first_run = False
            if np.isclose(cur_rank, 0.0):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'reach-100'))
                min_lr = learning_rates[max(lr_idx - 2, 0)]
                max_lr = learning_rates[lr_idx - 1] if \
                    learning_rates[lr_idx - 1] != min_lr else \
                    (min_lr + learning_rates[lr_idx]) / 2.
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Reached 100% non zero, new range: " +
                      f"{learning_rates}")
                lr_idx = 0
                cur_rank = -1
                continue
            # if len(rate_of_change) > 3 and \
            #         np.isclose(rate_of_change[-1], rate_of_change[-2],
            #                    atol=min_delta) and \
            #         np.isclose(rate_of_change[-2], rate_of_change[-3],
            #                    atol=min_delta):
            if np.less(rate_of_change[-1], min_delta):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'plateau'))
                exit_counter += 1
                min_lr = learning_rates[max(lr_idx - 2, 0)]
                max_lr = learning_rates[lr_idx]
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Hit Plateau, new range: " +
                      f"{learning_rates}")
                lr_idx = 0
                cur_rank = -1
                if exit_counter > exit_counter_thresh:
                    output_history.append(
                        (learning_rates[lr_idx],
                            cur_rank, 'exit-counter'))
                    break
                continue
        output_history.append((learning_rates[lr_idx],
                               cur_rank, 'nothing'))
        if exit_counter > 5:
            print(
                "LR Range Test Complete: Exit Counter: Final LR Range is " +
                f"{min_lr}-{max_lr}")
            output_history.append(
                (learning_rates[lr_idx], cur_rank, 'exit-counter'))
            break
        lr_idx += 1
    with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,rank,msg\n')
        for (lr, rank, msg) in output_history:
            f.write(f'{lr},{rank},{msg}\n')
    return learning_rates[-1]
