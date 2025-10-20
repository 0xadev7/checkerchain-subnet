import numpy as np

from checkerchain.rlhf.optim import project_to_simplex


def huber_grad(residual: float, delta: float) -> float:
    if abs(residual) <= delta:
        return residual
    return delta * (1.0 if residual > 0 else -1.0)


def online_update(
    w_prev: np.ndarray,
    batch_X_10: np.ndarray,  # (B,10)
    batch_y_100: np.ndarray,  # (B,)
    lr: float = 0.05,
    lambda_stability: float = 0.01,
    huber_delta: float = 1.0,
    recency_weights: np.ndarray | None = None,
    pgd_steps: int = 3,
) -> np.ndarray:
    w = w_prev.copy()
    X = np.asarray(batch_X_10, dtype=float)
    y = np.asarray(batch_y_100, dtype=float) / 10.0  # to 0..10

    if recency_weights is None:
        recency_weights = np.ones(X.shape[0], dtype=float)

    for _ in range(max(1, pgd_steps)):
        grad = np.zeros_like(w)
        w_dot_X = X @ w  # (B,)
        residuals = w_dot_X - y
        for i, (r, xi) in enumerate(zip(residuals, X)):
            g = huber_grad(float(r), huber_delta) * xi
            grad += recency_weights[i] * g

        grad /= recency_weights.sum() + 1e-9
        grad += 2.0 * lambda_stability * (w - w_prev)

        w = w - lr * grad
        w = project_to_simplex(w)

    return w
