from pathlib import Path
from typing import Callable, Union, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tap import Tap
from tqdm import tqdm

from echoes import ESNRegressor
from echoes.plotting import plot_predicted_ts
from echoes.utils import relu

from bioRNN.utils import (
    cv_generator,
    make_bioRRNs,
    clean_df_param_names,
    clean_col_names,
    label_W_df,
    assert_args_not_none,
)


class GridSamplingArguments(Tap):
    """Basic arguments that all experiments require"""
    n_samples: int = 10
    rand_partition: bool = True
    neuron_density: int = 4
    n_transient: int = 100
    n_jobs: int = -3

args = GridSamplingArguments().parse_args()

param_grid = {
    "spectral_radius": np.arange(91, 100, 2) * .01,
    "input_scaling": 10. ** np.arange(-6, 1),
    "leak_rate": [.6, .8, 1.],
    "activation_out": [relu],
    "n_transient": [args.n_transient],
}


def sample_grid(
    n_samples: Union[int, None],
    rand_partition: Union[bool, None],
    neuron_density: Union[int, None],
    param_grid: Union[Mapping, None] = None,
    data_maker: Union[Callable, None],
    scorer: Union[Dict, None] = None,
    directory_results: Union[Path, None],
    filename: Union[str, None],
    connectomes_path: Union[Path, None],
    connectomes: List[str] = ["macaque", "marmoset", "human"],
    n_jobs: int = -3,
):
    assert_args_not_none(locals().items())

    X, y = data_maker()


    results = pd.DataFrame()
    for connectome in tqdm(connectomes, desc="connectomes", total=len(connectomes), leave=True):
        results_connectome = []
        for _ in tqdm(range(n_samples), desc="samples", total=n_samples, leave=True):
            # Generate reservoirs based on connectomes
            Ws_map = make_bioRRNs(
                connectomes_path=connectomes_path,
                connectome_name=connectome,
                neuron_density=neuron_density,
                rand_partition=rand_partition
            )
            param_grid["W"] = list(Ws_map.values())

            # Sample grid of hyperparameters
            cv = cv_generator(X, test_size=.2)
            grid = GridSearchCV(
                ESNRegressor(),
                param_grid,
                scoring=scorer,
                cv=cv,
                n_jobs=n_jobs,
                refit=False,
                verbose=0,
            ).fit(X, y)

            # Save sample results
            results_sample = clean_col_names(pd.DataFrame(grid.cv_results_))
            results_sample = label_W_df(results_sample, Ws_map)
            results_sample["spectral_radius"] = np.round(results_sample.spectral_radius.values.astype(float), decimals=2)
            results_connectome.append(results_sample)

        # Save all samples results of a connectome
        results_connectome = pd.concat(results_connectome)
        results_connectome = create_fill_cols(results_connectome,
                                              [(0, "connectome", connectome),
                                               (1, "rand_partition", rand_partition),
                                               (2, "neuron_density", neuron_density)])

        results = pd.concat((results, results_connectome))
        results.to_csv(directory_results / filename, index=False)
