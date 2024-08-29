from typing import Mapping

import gin
import numpy as np

from echoes import ESNRegressor
import tasks

@gin.configurable
def evaluate_task(
    gridpoint: Mapping,
    task_params: Mapping,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_trials_exclude: int = 3,
) -> float:
    """
    Generate task train/test data, fit on train, evaluate on test.
    Return score.
    """
    return (
        ESNRegressor(
            n_transient=int(task_params["pattern_length"] * n_trials_exclude),
            **gridpoint
        )
        .fit(X_train, y_train)
        .score(X_test, y_test)
    )
