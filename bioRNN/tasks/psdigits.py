import numpy as np
from sklearn.datasets import load_digits


def make_X_y():
    X, y = load_digits(return_X_y=True)
    X /= X.max()
    perm = np.random.permutation(X.shape[1])
    X = X[:, perm]
    return X
