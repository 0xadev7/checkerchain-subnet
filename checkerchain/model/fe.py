from __future__ import annotations
from typing import Dict, List, Tuple, Optional
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
    # Derived features
    X_df["mean_score"] = X_df.mean(axis=1)
    X_df["std_score"] = X_df.std(axis=1)
    X_df["security_team"] = X_df["security"] * X_df["team"]
    X_df["utility_userbase"] = X_df["utility"] * X_df["userbase"]
    X_df["marketing_partnerships"] = X_df["marketing"] * X_df["partnerships"]
    X_df["clarity_roadmap"] = X_df["clarity"] * X_df["roadmap"]
    X_df["tokenomics_userbase"] = X_df["tokenomics"] * X_df["userbase"]
    X_df["security_roadmap"] = X_df["security"] * X_df["roadmap"]
    return X_df


def _drop_constant_columns(X_df: pd.DataFrame) -> pd.DataFrame:
    # Keep columns with any variance; constant cols can trigger no-gain leaves
    nunique = X_df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    # If all constant (edge case), fall back to original to avoid empty frame
    return X_df[keep] if len(keep) > 0 else X_df


def fe_transform_df(X_base: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    """Return engineered features as a DataFrame with stable, valid names."""
    X_df = _to_df(X_base)
    X_df = _drop_constant_columns(X_df)
    # Ensure column order = engineered list intersect available
    names_all = fe_names()
    names = [c for c in names_all if c in X_df.columns]
    return X_df.loc[:, names].astype(float), names


def fe_transform_matrix(X_base: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """(Backward-compat) Engineered features as ndarray + names."""
    X_df, names = fe_transform_df(X_base)
    return X_df.values.astype(float), names


def fe_transform_one(breakdown: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    x = np.array([[float(breakdown.get(k, 0.0)) for k in METRICS]], dtype=float)
    X, names = fe_transform_matrix(x)
    return X[0], names


def fe_transform_one_df(
    breakdown: Dict[str, float],
    feature_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Single-row engineered DataFrame, aligned to an optional saved feature order."""
    base = np.array([[float(breakdown.get(k, 0.0)) for k in METRICS]], dtype=float)
    X_df = _to_df(base)
    X_df = _drop_constant_columns(X_df)

    if feature_order:
        # Reindex to model’s saved order; fill any missing with 0.0
        # (in case training dropped constants that inference now provides)
        for col in feature_order:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df.loc[:, feature_order]
    else:
        names_all = fe_names()
        names = [c for c in names_all if c in X_df.columns]
        X_df = X_df.loc[:, names]

    return X_df.astype(float)
