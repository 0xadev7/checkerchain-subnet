import numpy as np
import cvxpy as cp


def fit_weights_bias_scale(
    X_10: np.ndarray,  # (N, D) features on ~0..10
    y_100: np.ndarray,  # (N,)    targets on 0..100
    huber_delta: float = 2.0,
    lambda_u_l2: float = 1e-3,  # L2 on u ( = b1*w )
    lambda_u_tv: float = 0.0,  # optional TV on adjacent entries of u
    cap_b1: float | None = None,  # optional upper bound on b1
):
    N, D = X_10.shape
    X = X_10.astype(float)
    y = y_100.astype(float)

    # Variables: u = b1 * w  (u >= 0) and intercept b0
    u = cp.Variable(D, nonneg=True)
    b0 = cp.Variable()

    pred = b0 + X @ u
    loss = cp.sum(cp.huber(pred - y, M=huber_delta))

    reg = lambda_u_l2 * cp.sum_squares(u)
    if lambda_u_tv > 0 and D > 1:
        reg += lambda_u_tv * cp.norm1(u[1:] - u[:-1])

    constraints = []
    if cap_b1 is not None:
        # Optional: enforce an upper bound on b1 (= sum(u)) to prevent over-scaling
        constraints.append(cp.sum(u) <= cap_b1)

    prob = cp.Problem(cp.Minimize(loss / N + reg), constraints)
    # Use any installed convex solver; ECOS, SCS, or OSQP (for quadratic objectives)
    prob.solve(solver=cp.ECOS, verbose=False)

    if u.value is None or b0.value is None:
        w_hat = np.ones(D) / D
        return w_hat, 0.0, 1.0, np.nan

    u_hat = np.maximum(u.value, 0.0)
    b1_hat = float(u_hat.sum())
    if b1_hat <= 1e-12:
        w_hat = np.ones(D) / D
        b1_hat = 0.0
    else:
        w_hat = u_hat / b1_hat

    return w_hat, float(b0.value), b1_hat, prob.value
