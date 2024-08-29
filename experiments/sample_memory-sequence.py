from functools import partial
from itertools import product
import os
from pathlib import Path

import numpy as np
from tap import Tap

from bioRNN.tasks.memory.sequence import make_X_y
from echoes.utils import relu

from bioRNN.utils import print_progress, CONNECTOMES_PATH, make_r2_scorer
from sample_task import sample_grid

from parameter_grids import best_params_memory_sequence, param_grid_memory_sequence
from concat_results_randpart_pattlen import (
    concat_hyperparams_memory_sequence,
    concat_evaluation_memory_sequence
)

# TODO: move Arguments classes away from sampling code?
class Arguments(Tap):
    """Arguments for connectome/task data generation and sampling"""
    n_samples: int = 10
    neuron_density: int
    n_transient: int = 100
    n_trials: int = 1000
    n_jobs: int = -2
    esn_params: str  # 'best' or 'full-grid'
    concatenate_results: str  # 'yes' or 'no'

args = Arguments().parse_args()

start, end = 5, 30
if args.esn_params == "best":
    param_grid = best_params_memory_sequence
    pattern_lengths = range(start, end)
elif args.esn_params == "full-grid":
    param_grid = param_grid_memory_sequence
    pattern_lengths = range(start, end, 5)
else:
    raise ValueError("esn_params must be 'best' or 'full-grid'")


rand_partitions = [True, False]
for rand_partition, pattern_length in product(rand_partitions, pattern_lengths):
    print_progress(rand_partition=rand_partition, pattern_length=pattern_length)

    data_maker = partial(
        make_X_y, pattern_length=pattern_length, n_trials=args.n_trials
    )

    sample_grid(
        data_maker=data_maker,
        rand_partition=rand_partition,
        param_grid=param_grid,
        directory_results=Path(os.environ["DATAICNS"])
        /"bioRNN/memory-sequence"
        /f"{'hyperparams' if args.esn_params == 'full-grid' else 'evaluation'}"
        /f"neuron-density-{args.neuron_density}",
        filename=f"rand-part-{rand_partition}_patt-len-{pattern_length}",
        n_samples=args.n_samples,
        neuron_density=args.neuron_density,
        scorer=make_r2_scorer(n_transient=args.n_transient),
        connectomes_path=CONNECTOMES_PATH,
        n_jobs=args.n_jobs,
    )

if args.concatenate_results == "yes":
    if args.esn_params == "full-grid":
        concat_hyperparams_memory_sequence(neuron_density=args.neuron_density)
    elif args.esn_params == "best":
        concat_evaluation_memory_sequence(start=start, end=end, neuron_density=args.neuron_density)
    else:
        raise ValueError("esn_params must be 'best' or 'full-grid'")
