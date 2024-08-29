from functools import partial
from itertools import product
import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from tap import Tap

from bioRNN.tasks.memory.capacity import make_X_y

from bioRNN.utils import print_progress, CONNECTOMES_PATH, make_memory_capacity_scorer
from sample_task import sample_grid
from parameter_grids import param_grid_memory_capacity, best_params_memory_capacity
from concat_results_randpart_pattlen import (
    concat_hyperparams_memory_capacity,
    concat_evaluation_memory_capacity,
)

# TODO: move Arguments classes away from sampling code?
class Arguments(Tap):
    """Arguments for connectome/task data generation and sampling"""
    n_samples: int = 10
    neuron_density: int = 4
    test_size: float = 0.2
    n_steps: int = 5_000
    n_jobs: int = -3
    esn_params: str  # 'best' or 'full-grid'
    concatenate_results: str  # 'yes' or 'no'


args = Arguments().parse_args()

if args.esn_params == "best":
    param_grid = best_params_memory_capacity
elif args.esn_params == "full-grid":
    param_grid = param_grid_memory_capacity
else:
    raise ValueError("esn_params must be 'best or 'full-grid'")


rand_partitions = [True, False]
#lags = range(5, 31)
lags = range(5, 101,)

data_maker = partial(
    make_X_y,
    make_X=partial(np.random.uniform, low=-0.5, high=0.5, size=args.n_steps),
    lags=lags,
)

for rand_partition in rand_partitions:
    print_progress(
        rand_partition=rand_partition,
    )

    sample_grid(
        data_maker=data_maker,
        rand_partition=rand_partition,
        param_grid=param_grid,
        directory_results=Path(os.environ["DATAICNS"])
        /"bioRNN/memory-capacity"
        #/f"{'hyperparams' if args.esn_params == 'full-grid' else 'evaluation'}"
        #/f"neuron-density-{args.neuron_density}",
        /f"reservoir-size-scaling",
        filename=f"rand-part-{rand_partition}_neuron-density-{args.neuron_density}",
        n_samples=args.n_samples,
        neuron_density=args.neuron_density,
        scorer=make_memory_capacity_scorer(n_transient=param_grid["n_transient"][0]),
        test_size=args.test_size,
        connectomes_path=CONNECTOMES_PATH,
        n_jobs=args.n_jobs,
    )

if args.concatenate_results == "yes":
    if args.esn_params == "full-grid":
        concat_hyperparams_memory_capacity(neuron_density=args.neuron_density)
    elif args.esn_params == "best":
        concat_evaluation_memory_capacity(neuron_density=args.neuron_density)
    else:
        raise ValueError("esn_params must be 'best' or 'full-grid'")
