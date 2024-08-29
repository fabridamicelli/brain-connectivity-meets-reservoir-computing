"""
Collection of function to post-process results: Collect experiments with different
pattern_lengths and merge them into one dataframe that is saved in the same directory
where the files are.
"""
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from bioRNN.utils import concat_results_randpart_pattlen

######################################################################################
# Memory sequential
######################################################################################


def concat_hyperparams_memory_sequence(*, neuron_density):
    dir_results = (
        Path(os.environ["DATAICNS"])
        / f"bioRNN/memory-sequence/hyperparams/neuron-density-{neuron_density}"
    )
    filename_base = ""
    df_concat = concat_results_randpart_pattlen(
        dir_results,
        filename_base,
        rand_partitions=[True, False],
        pattern_lengths=range(5, 31, 5),
    )
    df_concat.to_csv(dir_results/"hyperparams_all", index=False)


def concat_evaluation_memory_sequence(*, start, end, neuron_density):
    dir_results = (
        Path(os.environ["DATAICNS"])
        / f"bioRNN/memory-sequence/evaluation/neuron-density-{neuron_density}"
    )
    filename_base = ""
    df_concat = concat_results_randpart_pattlen(
        dir_results,
        filename_base,
        rand_partitions=[True, False],
        pattern_lengths=range(start, end),
    )
    df_concat.to_csv(dir_results/"all", index=False)


######################################################################################
# Memory capacity
######################################################################################


def concat_hyperparams_memory_capacity(*, neuron_density):
    dir_results = (
        Path(os.environ["DATAICNS"])
        / f"bioRNN/memory-capacity/hyperparams/neuron-density-{neuron_density}"
    )
    filename_base = ""
    df_concat = concat_results_randpart_pattlen(
        dir_results,
        filename_base,
        rand_partitions=[True, False],
        pattern_lengths=None,
    )
    df_concat.to_csv(dir_results/"hyperparams_all", index=False)


def concat_evaluation_memory_capacity(*, neuron_density, dir_results: Optional[Path] = None):
    if not dir_results:
        dir_results = (
            Path(os.environ["DATAICNS"])
            / f"bioRNN/memory-capacity/evaluation/neuron-density-{neuron_density}"
        )
    filename_base = ""
    df_concat = concat_results_randpart_pattlen(
        dir_results,
        filename_base,
        rand_partitions=[True, False],
        pattern_lengths=None,
    )
    df_concat.to_csv(dir_results/"all", index=False)


if __name__ == "__main__":
    # import sys
    # concat_evaluation_memory_sequence(start=5, end=31, neuron_density=1)
    # concat_evaluation_memory_sequence(int(sys.argv[1]), int(sys.argv[2]))
    # concat_hyperparams_memory_sequence(neuron_density=1)
    #concat_hyperparams_memory_capacity(neuron_density=1)
    concat_evaluation_memory_capacity(neuron_density=1)
