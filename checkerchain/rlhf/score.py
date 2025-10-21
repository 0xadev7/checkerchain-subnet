import numpy as np
from checkerchain.rlhf.constants import METRICS


def compute_overall_from_breakdown(
    breakdown: dict,
    w_simplex: list[float],
    beta0: float | None = None,
    beta1: float | None = None,
) -> float:
    x = np.array([float(breakdown.get(k, 0.0)) for k in METRICS], dtype=float)
    x = np.clip(x, 0.0, 10.0)
    w = np.asarray(w_simplex, dtype=float)
    w = w / (w.sum() + 1e-9)
    dot = float(np.dot(w, x))  # 0..10
    if beta0 is None:
        beta0 = 0.0
    if beta1 is None:
        beta1 = 1.0
    return float(np.clip(beta0 + beta1 * dot, 0.0, 100.0))  # calibrated
