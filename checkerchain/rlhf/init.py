import numpy as np
from scipy.optimize import nnls

from checkerchain.rlhf.optim import project_to_simplex
from checkerchain.rlhf.constants import N_METRICS, DEFAULT_W


def initialize_weights_nnls(X_10, y_100):
    """
    X_10: (N,10) breakdowns (0..10)
    y_100: (N,) trustScore (0..100)
    """
    if len(X_10) < 3:
        return np.array(DEFAULT_W, dtype=float)

    X = np.asarray(X_10, dtype=float)
    y = np.asarray(y_100, dtype=float) / 10.0  # to 0..10

    w_nnls, _ = nnls(X, y)  # w >= 0
    if (w_nnls <= 0).all():
        return np.array(DEFAULT_W, dtype=float)

    w = w_nnls / (w_nnls.sum() + 1e-9)  # sum to 1 (approx)
    return project_to_simplex(w)  # enforce exactly
