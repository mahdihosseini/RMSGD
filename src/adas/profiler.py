from typing import Tuple, List
from pstats import SortKey
from pathlib import Path

import cProfile
import pstats
import sys
import io

from memory_profiler import memory_usage

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .components import Statistics
    from .utils import pstats_to_dict
    from .gpu import GPU
else:
    from components import Statistics
    from utils import pstats_to_dict
    from gpu import GPU


class Profiler:
    base_mem_used = GPU(0).mem_used
    filename: Path = Path('stats.csv')

    def __init__(self, function):
        self.gpu = GPU(0)
        self.stream = None
        self.pr = cProfile.Profile()
        self.function = function
        self.statistics = List[Statistics]
        self.header_written = False
        self.trial = -1

    def __call__(self, trial, train_loader, test_loader, epoch: int,
                 device, optimizer, scheduler) -> Tuple[float, float]:
        if self.stream is None:
            self.stream = Profiler.filename.open('w+')
            print(f"AdaS: Profiler: Writing csv to {Profiler.filename}")
        else:
            if self.trial != trial:
                self.stream.close()
                self.stream = Profiler.filename.open('w+')
                print(f"AdaS: Profiler: Writing csv to {Profiler.filename}")
        self.trial = trial
        self.gpu.update()
        self.pr.enable()
        result = memory_usage(proc=(
            self.function,
            (trial, train_loader, test_loader, epoch, device, optimizer, scheduler)),
            max_usage=True, retval=True)
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats(
            "epoch_iteration|step|trial_iteration|test_main|evaluate")
        stats_list = pstats_to_dict(s.getvalue())
        header = 'epoch,epoch_gpu_mem_used,epoch_gpu_temp,epoch_ram_used'
        content = f'{epoch},{self.gpu.mem_used - Profiler.base_mem_used},{self.gpu.temperature},{result[0]}'
        for stat in stats_list:
            header += f",{stat['name']}_n_calls"
            header += f",{stat['name']}_tot_time"
            header += f",{stat['name']}_per_call1"
            header += f",{stat['name']}_cum_time"
            header += f",{stat['name']}_per_call2"
            content += f",{stat['n_calls']}"
            content += f",{stat['tot_time']}"
            content += f",{stat['per_call1']}"
            content += f",{stat['cum_time']}"
            content += f",{stat['per_call2']}"
        header += '\n'
        content += '\n'
        if not self.header_written:
            self.stream.write(header)
            self.header_written = True
        self.stream.write(content)
        return result[1]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stream.close()
