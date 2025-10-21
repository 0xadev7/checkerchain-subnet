from __future__ import annotations
import io, os, time, math, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
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


def load_xy_full() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


def remove_outliers(X: np.ndarray, y: np.ndarray, low=1, high=99):
    if len(y) < 50:
        return X, y
    q1, q99 = np.percentile(y, [low, high])
    mask = (y >= q1) & (y <= q99)
    return X[mask], y[mask]


def sample_param() -> Dict[str, Any]:
    return {
        "objective": "mae",
        "learning_rate": 10 ** random.uniform(-2.0, -0.9),
        "n_estimators": random.randint(600, 2000),
        "num_leaves": random.randint(16, 64),
        "max_depth": random.choice([-1, 4, 5, 6, 7, 8]),
        "min_child_samples": random.randint(8, 80),
        "subsample": random.uniform(0.7, 1.0),
        "colsample_bytree": random.uniform(0.7, 1.0),
        "reg_lambda": 10 ** random.uniform(-3, 1),
        "reg_alpha": 10 ** random.uniform(-3, 1),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        # Silence LightGBM logs/warnings (including "No further splits...")
        "verbosity": -1,
        # Small safety to avoid pathological histogram shapes in tiny data
        "force_col_wise": True,
    }


def cv_score_params(
    X_df: pd.DataFrame, y: np.ndarray, params: Dict[str, Any]
) -> Tuple[float, float]:
    """Return (cv_mae, std_mae) using KFold with early stopping."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    maes: List[float] = []

    # Use index-based splits while keeping DataFrame (with column names)
    for tr_idx, va_idx in kf.split(X_df.values, y):
        X_tr = X_df.iloc[tr_idx]
        X_va = X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),  # silence eval logs
            ],
        )
        y_hat = model.predict(X_va, num_iteration=model.best_iteration_)
        maes.append(mean_absolute_error(y_va, y_hat))

    return float(np.mean(maes)), float(np.std(maes))


def train_and_register():
    ensure_indexes()
    X_base, y = load_xy_full()
    if len(y) < 100:
        bt.logging.info(f"Not enough rows to train (have {len(y)}).")
        return

    X_base, y = remove_outliers(X_base, y, low=1, high=99)

    # Feature engineering as DataFrame with column names
    X_df, feat_names = fe_transform_df(X_base)

    # Parameter search
    best = {"mae": 1e9, "std": 0.0, "params": None}
    for i in range(N_PARAM_SAMPLES):
        params = sample_param()
        cv_mae, cv_std = cv_score_params(X_df, y, params)
        bt.logging.info(
            f"[{i+1:02d}/{N_PARAM_SAMPLES}] MAE={cv_mae:.3f} (±{cv_std:.3f}) params={params}"
        )
        if cv_mae < best["mae"]:
            best = {"mae": cv_mae, "std": cv_std, "params": params}

    bt.logging.info(
        f"\nBest CV MAE={best['mae']:.3f} (±{best['std']:.3f}) with params:"
    )
    bt.logging.info(best["params"])

    # Final refit with hold-out
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.15, random_state=RANDOM_STATE
    )
    final = lgb.LGBMRegressor(**best["params"])
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
        "feature_order": feat_names,  # columns used after dropping constants
        "base_feature_order": METRICS,  # original 10 for sanity
        "metrics": {
            "cv_mae": float(best["mae"]),
            "cv_std": float(best["std"]),
            "val_mae": val_mae,
            "n_rows": int(len(y)),
            "n_splits": N_SPLITS,
            "best_iteration": int(getattr(final, "best_iteration_", 0) or 0),
        },
        "params": best["params"],
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
