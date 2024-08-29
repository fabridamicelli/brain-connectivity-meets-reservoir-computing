"""
Evaluate different constellations of echo state network parameters on a task for all connectomes.

bio2art is used to create reservoirs from connectomes.
Only connectomes_path, connectome, neuron_density and rand_partition
are treated here as variables. The rest of parameters used to create
the reservoirs are just the defaults (see function make_bioRRNs for details).
"""
import inspect
from pathlib import Path
from typing import Callable, Union, Mapping, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from echoes import ESNRegressor

from bioRNN.utils import (
    cv_generator,
    make_bioRRNs,
    clean_col_names,
    label_W_df,
    create_fill_cols,
    assert_args_not_none,
)


def sample_grid(*,
    n_samples: int,
    rand_partition: bool,
    neuron_density: int,
    param_grid: Mapping,
    data_maker: Callable,
    test_size: Union[int, float] = 0.2,
    scorer: Dict,
    directory_results: Path,
    filename: str,
    connectomes_path: Path,
    connectome_names: Tuple[str] = ("macaque", "marmoset", "human"),
    estimator_class: Union[ESNRegressor, None] = ESNRegressor,  # TODO ESNImageClassifier
    n_jobs: int = -3,
) -> None:
    assert_args_not_none(locals().items())
    # Grab parameters to pass to sample_grid_point function
    params_grid_point = {
        k: v
        for k, v in locals().items()
        if k in inspect.signature(sample_grid_point).parameters
    }

    X, y = data_maker()

    results = pd.DataFrame()
    for connectome_name in tqdm(
        connectome_names, desc="connectomes", total=len(connectome_names)
    ):
        params_grid_point["connectome_name"] = connectome_name
        results_connectome = sample_grid_point(X=X, y=y, **params_grid_point)
        results = pd.concat((results, results_connectome))
        results.to_csv(directory_results / filename, index=False)


def sample_grid_point(*,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    param_grid: Mapping,
    scorer: Dict,
    test_size: Union[int, float] = 0.2,
    connectomes_path: Path,
    connectome_name: str,
    rand_partition: bool,
    neuron_density: int,
    estimator_class: Union[ESNRegressor, None] = ESNRegressor,  # TODO: ESNImageClassifier
    n_jobs: int = -3,
) -> pd.DataFrame:
    """Return DataFrame with results of evaluating the grid point on n_samples times"""
    assert_args_not_none(locals().items())
    results_connectome = []
    for _ in tqdm(range(n_samples), desc="samples", total=n_samples):
        # Generate reservoirs based on connectomes
        Ws_map = make_bioRRNs(
            connectomes_path=connectomes_path,
            connectome_name=connectome_name,
            neuron_density=neuron_density,
            rand_partition=rand_partition,
        )
        param_grid["W"] = list(Ws_map.values())

        cv = cv_generator(X, test_size=test_size)
        grid = GridSearchCV(
            estimator_class(),
            param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=n_jobs,
            refit=False,
            verbose=0,
        ).fit(X, y)

        # Save partial results
        results_sample = clean_col_names(pd.DataFrame(grid.cv_results_))
        results_sample = label_W_df(results_sample, Ws_map)
        results_sample["spectral_radius"] = np.round(
            results_sample.spectral_radius.values.astype(float), decimals=2
        )
        results_connectome.append(results_sample)

    results_connectome = pd.concat(results_connectome)
    results_connectome = create_fill_cols(
        results_connectome,
        [
            (0, "connectome", connectome_name),
            (1, "rand_partition", rand_partition),
            (2, "neuron_density", neuron_density),
        ],
    )

    return results_connectome
