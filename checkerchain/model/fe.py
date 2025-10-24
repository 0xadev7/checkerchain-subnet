from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re

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
    Returns (engineered_features_df, feature_names) with sanitized column names.
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

    # Matrices for vectorized ops
    core_mat = df[core_cols].to_numpy(dtype=float)
    conf_mat = df[conf_cols].to_numpy(dtype=float)

    feat: Dict[str, np.ndarray] = {}

    # Per-dimension transforms
    for i in range(10):
        ccol = f"X[{i}]"
        fcol = f"X[{10+i}]"
        core = core_mat[:, i]
        conf = conf_mat[:, i]
        conf2 = conf * conf
        core2 = core * core

        feat[f"{ccol}_core"] = core
        feat[f"{fcol}_conf"] = conf
        feat[f"{ccol}_w1"] = core * conf
        feat[f"{ccol}_w2"] = core * conf2
        feat[f"{ccol}_quad_w"] = core2 * conf
        feat[f"{ccol}_over_conf"] = core / (conf + EPS)
        feat[f"{ccol}_minus_10conf"] = core - 10.0 * conf

    # Global aggregates
    feat["core_mean"] = core_mat.mean(axis=1)
    feat["core_std"] = core_mat.std(axis=1)
    feat["core_range"] = core_mat.max(axis=1) - core_mat.min(axis=1)
    feat["core_sum"] = core_mat.sum(axis=1)
    feat["conf_mean"] = conf_mat.mean(axis=1)
    feat["conf_std"] = conf_mat.std(axis=1)
    feat["conf_sum"] = conf_mat.sum(axis=1)
    wcore = core_mat * conf_mat
    feat["wcore_mean"] = wcore.mean(axis=1)
    feat["wcore_std"] = wcore.std(axis=1)

    # Pairwise weighted interactions
    for i in range(10):
        for j in range(i + 1, 10):
            feat[f"int_w_{i}_{j}"] = (
                core_mat[:, i] * core_mat[:, j] * conf_mat[:, i] * conf_mat[:, j]
            )

    # Mild nonlinearity
    for i in range(5):
        feat[f"core2_conf2_{i}"] = (core_mat[:, i] ** 2) * (conf_mat[:, i] ** 2)

    # ---- sanitize names once, then build DataFrame in one shot ----
    orig_cols = list(feat.keys())

    def _sanitize(name: str) -> str:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", name)  # replace [, ], etc
        s = re.sub(r"__+", "_", s).strip("_")  # collapse underscores
        return s

    new_cols: List[str] = []
    seen = {}
    for c in orig_cols:
        sc = _sanitize(c)
        if sc in seen:
            seen[sc] += 1
            sc = f"{sc}__{seen[sc]}"
        else:
            seen[sc] = 0
        new_cols.append(sc)

    feats = pd.DataFrame(
        {nc: feat[oc] for oc, nc in zip(orig_cols, new_cols)},
        index=df.index,
        dtype=float,
    )
    return feats, new_cols


# ----------------------------
# Public FE API used by train/inference
# ----------------------------
def fe_transform_df_from_raw(X_base: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    if X_base.ndim != 2 or X_base.shape[1] != 20:
        raise ValueError(f"Expected X_base shape (n, 20); got {X_base.shape}")
    raw = pd.DataFrame(X_base, columns=[f"X[{i}]" for i in range(20)])
    feats, feat_names = build_features_from_raw_X(raw)
    mask = np.isfinite(feats.values).all(axis=1)
    feats = feats.loc[mask].astype(float)
    return feats, feat_names


def fe_transform_one_df_from_breakdown(
    breakdown: Dict[str, float],
    confidence: Dict[str, float],
    feature_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    core_vals = [float(breakdown.get(m, 0.0)) for m in METRICS[:10]]
    conf_vals = [float(confidence.get(m, 0.5)) for m in METRICS[:10]]
    raw = np.array([core_vals + conf_vals], dtype=float)

    raw_df = pd.DataFrame(raw, columns=[f"X[{i}]" for i in range(20)])
    feats, feat_names = build_features_from_raw_X(raw_df)  # already sanitized

    if feature_order:
        # Create any missing columns (robust to future FE changes)
        for col in feature_order:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats.loc[:, feature_order]
    else:
        feats = feats.loc[:, feat_names]

    return feats.astype(float)
