from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# If you still keep METRICS for downstream mapping, import it;
# it's used only for inference mapping (breakdown/confidence -> raw X)
from checkerchain.database.mongo import METRICS

CORE_IDX = list(range(0, 10))
CONF_IDX = list(range(10, 20))
EPS = 1e-6


# ----------------------------
# Core helpers from train_ex.py
# ----------------------------
def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_features_from_raw_X(raw_X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expects raw_X to have columns: X[0]..X[19]
      - X[0..9]   = core (scores)
      - X[10..19] = confidence (0..1)
    Returns (engineered_features_df, feature_names)
    """
    expected_cols = [f"X[{i}]" for i in range(20)]
    missing = [c for c in expected_cols if c not in raw_X.columns]
    if missing:
        raise ValueError(f"X missing columns: {missing}")

    df = raw_X[expected_cols].copy()
    core_cols = [f"X[{i}]" for i in CORE_IDX]
    conf_cols = [f"X[{i}]" for i in CONF_IDX]

    df = _ensure_numeric(df, core_cols + conf_cols)
    df[core_cols] = df[core_cols].clip(lower=-5, upper=15)
    df[conf_cols] = df[conf_cols].clip(lower=0, upper=1)

    feats = pd.DataFrame(index=df.index)

    # Per-dimension transforms
    for i in range(10):
        ccol = f"X[{i}]"
        fcol = f"X[{10+i}]"
        core, conf = df[ccol], df[fcol]
        conf2, core2 = conf**2, core**2

        feats[f"{ccol}_core"] = core
        feats[f"{fcol}_conf"] = conf
        feats[f"{ccol}_w1"] = core * conf
        feats[f"{ccol}_w2"] = core * conf2
        feats[f"{ccol}_quad_w"] = core2 * conf
        feats[f"{ccol}_over_conf"] = core / (conf + EPS)
        feats[f"{ccol}_minus_10conf"] = core - 10.0 * conf

    # Global aggregates
    core = df[core_cols]
    conf = df[conf_cols]
    feats["core_mean"] = core.mean(axis=1)
    feats["core_std"] = core.std(axis=1)
    feats["core_range"] = core.max(axis=1) - core.min(axis=1)
    feats["core_sum"] = core.sum(axis=1)
    feats["conf_mean"] = conf.mean(axis=1)
    feats["conf_std"] = conf.std(axis=1)
    feats["conf_sum"] = conf.sum(axis=1)
    weighted_core = core.values * conf.values
    feats["wcore_mean"] = weighted_core.mean(axis=1)
    feats["wcore_std"] = pd.DataFrame(weighted_core, index=df.index).std(axis=1)

    # Pairwise weighted interactions
    for i in range(10):
        for j in range(i + 1, 10):
            feats[f"int_w_{i}_{j}"] = (
                df[f"X[{i}]"] * df[f"X[{j}]"] * df[f"X[{10+i}]"] * df[f"X[{10+j}]"]
            )

    # Mild nonlinearity
    for i in range(5):
        feats[f"core2_conf2_{i}"] = (df[f"X[{i}]"] ** 2) * (df[f"X[{10+i}]"] ** 2)

    return feats, list(feats.columns)


# ----------------------------
# Public FE API used by train/inference
# ----------------------------
def fe_transform_df_from_raw(X_base: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    """
    X_base: ndarray of shape (n, 20) matching the raw layout described above.
    """
    if X_base.ndim != 2 or X_base.shape[1] != 20:
        raise ValueError(f"Expected X_base shape (n, 20); got {X_base.shape}")
    raw = pd.DataFrame(X_base, columns=[f"X[{i}]" for i in range(20)])
    feats, feat_names = build_features_from_raw_X(raw)
    # keep only finite rows (should already be)
    mask = np.isfinite(feats.values).all(axis=1)
    feats = feats.loc[mask].astype(float)
    return feats, feat_names


def fe_transform_one_df_from_breakdown(
    breakdown: Dict[str, float],
    confidence: Dict[str, float],
    feature_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert metric dicts into raw X[0..19], engineer features, and align to feature_order.
    - breakdown: {metric -> score}
    - confidence: {metric -> confidence in [0,1]}
    """
    # Map METRICS → positions X[0..9] and X[10..19]
    core_vals = [float(breakdown.get(m, 0.0)) for m in METRICS[:10]]
    conf_vals = [
        float(confidence.get(m, 0.5)) for m in METRICS[:10]
    ]  # sensible default
    raw = np.array([core_vals + conf_vals], dtype=float)

    raw_df = pd.DataFrame(raw, columns=[f"X[{i}]" for i in range(20)])
    feats, feat_names = build_features_from_raw_X(raw_df)

    if feature_order:
        for col in feature_order:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats.loc[:, feature_order]
    else:
        feats = feats.loc[:, feat_names]

    return feats.astype(float)
