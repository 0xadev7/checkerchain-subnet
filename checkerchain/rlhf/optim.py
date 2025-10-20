import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto {w | w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(1, v.size + 1)))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)
