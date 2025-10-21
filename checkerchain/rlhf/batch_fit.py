import numpy as np
import cvxpy as cp


def fit_weights_bias_scale(
    X_10: np.ndarray,  # (N,10) breakdowns on 0..10
    y_100: np.ndarray,  # (N,)   trustScore on 0..100
    huber_delta: float = 2.0,  # on 0..100 scale (≈2-4 works well)
    lambda_w_l2: float = 1e-3,  # small smoothing on w
    lambda_w_tv: float = 0.0,  # optional total variation on adjacent weights
):
    N, D = X_10.shape
    X = X_10.astype(float)
    y = y_100.astype(float)

    w = cp.Variable(D, nonneg=True)
    b0 = cp.Variable()  # intercept β0
    b1 = cp.Variable(nonneg=True)  # scale β1 ≥ 0

    pred = b0 + b1 * (X @ w) * 10.0 / 10.0  # keep unit-consistent, (X@w) is 0..10
    loss = cp.sum(cp.huber(pred - y, M=huber_delta))

    reg = lambda_w_l2 * cp.sum_squares(w)
    if lambda_w_tv > 0 and D > 1:
        reg += lambda_w_tv * cp.norm1(w[1:] - w[:-1])

    constraints = [cp.sum(w) == 1]  # simplex sum
    prob = cp.Problem(cp.Minimize(loss / N + reg), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    # Fallbacks if solve failed
    if w.value is None or b0.value is None or b1.value is None:
        w_hat = np.ones(D) / D
        return w_hat, 0.0, 1.0, np.nan

    w_hat = np.maximum(w.value, 0.0)
    w_hat = w_hat / (w_hat.sum() + 1e-12)
    return w_hat, float(b0.value), max(0.0, float(b1.value)), prob.value
