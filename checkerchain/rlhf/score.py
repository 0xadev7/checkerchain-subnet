import numpy as np

from checkerchain.rlhf.constants import METRICS


def compute_overall_from_breakdown(breakdown: dict, w_simplex: list[float]) -> float:
    x = np.array([float(breakdown.get(k, 0.0)) for k in METRICS], dtype=float)
    x = np.clip(x, 0.0, 10.0)
    w = np.asarray(w_simplex, dtype=float)
    w = w / (w.sum() + 1e-9)
    return float(10.0 * np.dot(w, x))  # 0..100
