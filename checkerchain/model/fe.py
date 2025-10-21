from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from checkerchain.database.mongo import METRICS


def fe_names() -> List[str]:
    extra = [
        "mean_score",
        "std_score",
        "security_team",
        "utility_userbase",
        "marketing_partnerships",
        "clarity_roadmap",
        "tokenomics_userbase",
        "security_roadmap",
    ]
    return METRICS + extra


def _to_df(X_base: np.ndarray) -> pd.DataFrame:
    X_df = pd.DataFrame(X_base, columns=METRICS).astype(float)
    # Derived
    X_df["mean_score"] = X_df.mean(axis=1)
    X_df["std_score"] = X_df.std(axis=1)
    X_df["security_team"] = X_df["security"] * X_df["team"]
    X_df["utility_userbase"] = X_df["utility"] * X_df["userbase"]
    X_df["marketing_partnerships"] = X_df["marketing"] * X_df["partnerships"]
    X_df["clarity_roadmap"] = X_df["clarity"] * X_df["roadmap"]
    X_df["tokenomics_userbase"] = X_df["tokenomics"] * X_df["userbase"]
    X_df["security_roadmap"] = X_df["security"] * X_df["roadmap"]
    return X_df


def fe_transform_matrix(X_base: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    X_df = _to_df(X_base)
    names = fe_names()
    return X_df[names].values.astype(float), names


def fe_transform_one(breakdown: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    x = np.array([[float(breakdown.get(k, 0.0)) for k in METRICS]], dtype=float)
    X, names = fe_transform_matrix(x)
    return X[0], names
