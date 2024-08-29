"""
Sample hyperparameters of echo state network solving the sequence memory task.
"""
from itertools import product
import os
from pathlib import Path
from typing import Dict

from hyperparams_grid_task_base import sample_grid, GridSamplingArguments
from bioRNN.utils import CONNECTOMES_PATH, make_r2_scorer, print_progress
from bioRNN.tasks.memory.sequence import (
    make_X_y,
    score_task,
    make_sklearn_scorer
)


class Arguments(Tap):
    """Arguments for connectome/task data generation and sampling"""
    n_samples: int = 10
    rand_partition: bool = True
    neuron_density: int = 4
    pattern_length: int = 5
    n_trials: int = 1000
    n_jobs: int = -2
args = Arguments().parse_args()


if __name__ == "__main__":
    from functools import partial
    from datetime import datetime as dt

    # Loop through task parameters constellations
    rand_partitions = [True, False]
    pattern_lengths = range(5, 31, 5)
    for rand_partition, pattern_length in product(rand_partitions, pattern_lengths):
        print_progress(rand_partition=rand_partition, pattern_length=pattern_length)

        data_maker = partial(make_X_y, pattern_length=args.pattern_length, n_trials=args.n_trials)
        sample_grid(
            data_maker=data_maker,
            rand_partition=rand_partition,
            filename=f"hyperparams_rand-part-{rand_partition}_patt-len-{pattern_length}",
            directory_results=Path(os.environ["DATAICNS"]) / "bioRNN/memory-sequential/results",

            n_transient=args.n_transient,
            n_samples=args.n_samples,
            neuron_density=args.neuron_density,
            scorer_maker=partial(make_r2_scorer, n_transient=args.n_transient),
            connectomes_path=CONNECTOMES_PATH,
            n_jobs=args.n_jobs,
        )
