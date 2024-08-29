"""
Memory Capacit task as in defined by H. Jaeger in
"Short Term Memory in Echo State Networks" (2001).
"""
from functools import partial
from typing import Callable, Dict, List, Union, Tuple

import numpy as np


def make_X_y(
    make_X: Callable,
    lags: np.ndarray,
    cut: int = 0,
) -> np.ndarray:
    """
    make_X: Callable
        Function to generate inputs. Should return a 1D np.ndarray.
        Note that the size of the output of make_X determines the size
        of X used for the task.
        Example:
        from functools import partial
        make_X = partial(np.random.uniform, low=-.5, high=.5, size=200)
    lags: np.ndarray
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).
    cut: int, optional
        Number of initial steps to cut out.
        Make be at least larger than max(lags) if you want to avoid circle sequence.
    """
    X = make_X()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = _make_lagged_y(X, lags=lags)
    return X[cut:], y[cut:]

def _make_lagged_y(inputs: np.ndarray, lags: np.ndarray,) -> np.ndarray:
    """
    Generate delayed versions of inputs sequence.
    One sequence is generated for each lag value.

    Parameters
    ----------
    inputs: np.ndarray
        Signal to lag. It will be flattened before lagging,
        as it is supposed to be only one input chanel.
    lags: np.ndarray
        Delays to be evaluated (memory capacity).
        For example: np.arange(1, 31, 5).

    Returns
    -------
    lagged_inputs: np.ndarray of shape (len(inputs), len(lags))
        Array of lagged version of the inputs sequence.
        Each column represents U(t-k), where k is the lag.

    Examples
    --------
    >>> inputs = np.arange(5)
    >>> make_lagged_y(inputs, [1, 3])
    >>> array([[4., 2.],
               [0., 3.],
               [1., 4.],
               [2., 0.],
               [3., 1.]])

    >>> inputs = np.arange(5)
    >>> make_lagged_y(inputs, [1, 3])
    >>> array([[1., 4.],
               [2., 0.],
               [3., 1.]])
    """
    assert isinstance(inputs, np.ndarray), "inputs must be np.ndarray"
    inputs = inputs.flatten()
    inputs_lagged = np.zeros((len(inputs), len(lags)))
    for col, lag in enumerate(lags):
        inputs_lagged[:, col] = np.roll(inputs, lag)
    return inputs_lagged


def forgetting(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List, float]:
    """
    Scoring function.
    Return forgetting curve (r2(k) for all k) and Memory capacity
    (sum over all values of the curve).

    y_pred and y_true are compared column-wise,
    assuming each column contains the values for a
    give delay.
    """
    assert y_pred.shape == y_true.shape, "y_pred and y_true must have same shape"
    assert y_pred.shape[0] > 1, "Error while computing forgetting: y_pred has less than 1 sample. Increase the number of samples (steps)"
    r2s = []
    for true, pred in zip(y_true.T, y_pred.T):
        r2 = np.corrcoef(true, pred)[0, 1]
        r2s.append(0 if r2 is None else r2 ** 2)
    return r2s, np.sum(r2s)
