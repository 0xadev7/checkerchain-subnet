from __future__ import annotations
import io, time, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import bittensor as bt

from checkerchain.database.mongo import (
    dataset_col,
    models_col,
    ensure_indexes,
    METRICS,
)
from checkerchain.model.fe import fe_transform_df, fe_names

RANDOM_STATE = 42
N_SPLITS = 5
N_PARAM_SAMPLES = 30
EARLY_STOP = 100
PRUNE_RESIDUAL_OUTLIERS = True  # set True to drop worst ~2% label outliers


def load_xy_full() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


def remove_value_outliers(X: np.ndarray, y: np.ndarray, low=1, high=99):
    if len(y) < 50:
        return X, y
    q1, q99 = np.percentile(y, [low, high])
    mask = (y >= q1) & (y <= q99)
    return X[mask], y[mask]


def sample_param() -> Dict[str, Any]:
    return {
        # USE L2 here so monotone_constraints are allowed
        "objective": "regression",  # L2 (supports monotone_constraints)
        "learning_rate": 10 ** random.uniform(-2.0, -1.0),  # 0.01..0.1
        "n_estimators": random.randint(800, 2200),
        "num_leaves": random.randint(16, 64),
        "max_depth": random.choice([-1, 4, 5, 6, 7, 8]),
        "min_child_samples": random.randint(16, 100),
        "min_gain_to_split": 1e-6,
        "subsample": random.uniform(0.7, 1.0),
        "colsample_bytree": random.uniform(0.7, 1.0),
        "reg_lambda": 10 ** random.uniform(-3, 1),
        "reg_alpha": 10 ** random.uniform(-3, 1),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }


def build_monotone_constraints(cols: List[str]) -> List[int]:
    # Domain prior: most metrics/combos should be positively related,
    # dispersion (std_score) should be negative.
    positive = {
        # base
        "project",
        "userbase",
        "utility",
        "security",
        "team",
        "tokenomics",
        "marketing",
        "roadmap",
        "clarity",
        "partnerships",
        # engineered positives
        "mean_score",
        "security_team",
        "utility_userbase",
        "marketing_partnerships",
        "clarity_roadmap",
        "tokenomics_userbase",
        "security_roadmap",
    }
    negative = {"std_score"}
    mc: List[int] = []
    for c in cols:
        if c in positive:
            mc.append(1)
        elif c in negative:
            mc.append(-1)
        else:
            mc.append(0)
    return mc


def cv_score_params(
    X_df: pd.DataFrame,
    y: np.ndarray,
    params: Dict[str, Any],
    monotone_constraints: List[int],
) -> Tuple[float, float]:
    """Return (cv_mae, std_mae) using StratifiedKFold (binned y) + early stopping."""
    params_local = params.copy()
    params_local["monotone_constraints"] = monotone_constraints

    # Bin labels (0..100) into ~21 buckets for stratification
    y_bins = np.clip((y / 5).astype(int), 0, 20)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    maes: List[float] = []
    for tr_idx, va_idx in kf.split(X_df.values, y_bins):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params_local)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        y_hat = model.predict(X_va, num_iteration=model.best_iteration_)
        maes.append(mean_absolute_error(y_va, y_hat))

    return float(np.mean(maes)), float(np.std(maes))


def prune_residual_outliers(X_df: pd.DataFrame, y: np.ndarray, pct: float = 2.0):
    """Optional: quick MAE model, drop top pct% largest |residuals|, return pruned."""
    if len(y) < 300 or pct <= 0.0:
        return X_df, y
    warm = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        verbosity=-1,
        force_col_wise=True,
        random_state=RANDOM_STATE,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE
    )
    warm.fit(X_tr, y_tr)
    res = np.abs(warm.predict(X_df) - y)
    thr = np.percentile(res, 100 - pct)
    mask = res <= thr
    kept = int(mask.sum())
    bt.logging.info(f"[prune] Dropping top {pct:.1f}% residuals → keep {kept}/{len(y)}")
    return X_df.loc[mask], y[mask]


def train_and_register():
    ensure_indexes()
    X_base, y = load_xy_full()
    if len(y) < 100:
        bt.logging.info(f"Not enough rows to train (have {len(y)}).")
        return

    # Gentle value trimming
    X_base, y = remove_value_outliers(X_base, y, low=1, high=99)

    # Feature engineering as DataFrame with names
    X_df, feat_names = fe_transform_df(X_base)

    # Optional: residual-based pruning (OFF by default)
    if PRUNE_RESIDUAL_OUTLIERS:
        X_df, y = prune_residual_outliers(X_df, y, pct=2.0)

    # Monotone constraints from engineered names actually present
    monotone_constraints = build_monotone_constraints(feat_names)

    # Parameter search
    best = {"mae": 1e9, "std": 0.0, "params": None}
    for i in range(N_PARAM_SAMPLES):
        params = sample_param()
        cv_mae, cv_std = cv_score_params(X_df, y, params, monotone_constraints)
        bt.logging.info(
            f"[{i+1:02d}/{N_PARAM_SAMPLES}] MAE={cv_mae:.3f} (±{cv_std:.3f}) params={params}"
        )
        if cv_mae < best["mae"]:
            best = {"mae": cv_mae, "std": cv_std, "params": params}

    bt.logging.info(
        f"\nBest CV MAE={best['mae']:.3f} (±{best['std']:.3f}) with params:"
    )
    bt.logging.info(best["params"])

    # Final refit with a hold-out split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.15, random_state=RANDOM_STATE
    )
    final_params = best["params"].copy()
    final_params["monotone_constraints"] = monotone_constraints

    final = lgb.LGBMRegressor(**final_params)
    final.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    y_hat = final.predict(X_va, num_iteration=final.best_iteration_)
    val_mae = float(mean_absolute_error(y_va, y_hat))

    # Serialize
    import joblib, io

    buf = io.BytesIO()
    joblib.dump(final, buf)
    model_blob = buf.getvalue()

    # Save to registry
    doc = {
        "createdAt": time.time(),
        "algo": "lightgbm.LGBMRegressor",
        "feature_order": feat_names,  # engineered columns used
        "base_feature_order": METRICS,  # original 10 for sanity
        "metrics": {
            "cv_mae": float(best["mae"]),
            "cv_std": float(best["std"]),
            "val_mae": val_mae,
            "n_rows": int(len(y)),
            "n_splits": N_SPLITS,
            "best_iteration": int(getattr(final, "best_iteration_", 0) or 0),
        },
        "params": final_params,
        "joblib_blob": model_blob,
    }
    models_col.insert_one(doc)
    bt.logging.info(
        f"[OK] Stored tuned model. CV_MAE={best['mae']:.3f}, VAL_MAE={val_mae:.3f}, rows={len(y)}"
    )


if __name__ == "__main__":
    bt.logging.set_trace()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    train_and_register()
