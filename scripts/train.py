from __future__ import annotations
import io, time, random, json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import bittensor as bt

from checkerchain.database.mongo import (
    dataset_col,
    models_col,
    ensure_indexes,
    METRICS,
)

# Confidence-aware FE
from checkerchain.model.fe import fe_transform_df_from_raw

RANDOM_STATE = 42
N_SPLITS = 5
N_PARAM_SAMPLES = 30  # only used when trials=0 (random search)
EARLY_STOP = 100
PRUNE_RESIDUAL_OUTLIERS = False  # optional


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
    # Random-search space (used when trials=0)
    return {
        "objective": "regression",
        "learning_rate": 10 ** random.uniform(-2.0, -1.1),  # ~0.01..0.079
        "n_estimators": random.randint(1500, 3500),
        "num_leaves": random.randint(31, 96),
        "max_depth": random.choice([-1, 6, 8, 10]),
        "min_child_samples": random.randint(10, 80),
        "min_gain_to_split": 1e-6,
        "subsample": random.uniform(0.7, 0.95),
        "colsample_bytree": random.uniform(0.7, 1.0),
        "reg_lambda": 10 ** random.uniform(-3, 0.7),
        "reg_alpha": 10 ** random.uniform(-3, 0.7),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }


def make_strat_bins(y: np.ndarray, n_splits: int, max_bins: int = 10) -> np.ndarray:
    y = np.asarray(y)
    n_bins = min(max_bins, max(3, len(np.unique(y)) // n_splits))
    n_bins = max(3, n_bins)
    while n_bins >= 3:
        try:
            q = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
            counts = np.bincount(q)
            if (counts >= n_splits).all():
                return q
        except Exception:
            pass
        n_bins -= 1
    return np.zeros_like(y, dtype=int)


def cv_with_oof_and_params(
    X_df: pd.DataFrame,
    y: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, float, np.ndarray]:
    y_bins = make_strat_bins(y, N_SPLITS, max_bins=10)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros_like(y, dtype=float)
    maes: List[float] = []

    for tr_idx, va_idx in kf.split(X_df.values, y_bins):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        p = m.predict(X_va, num_iteration=m.best_iteration_)
        oof_pred[va_idx] = p
        maes.append(mean_absolute_error(y_va, p))

    return float(np.mean(maes)), float(np.std(maes)), oof_pred


def prune_residual_outliers(X_df: pd.DataFrame, y: np.ndarray, pct: float = 2.0):
    if len(y) < 300 or pct <= 0.0:
        return X_df, y
    warm = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
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


# ---------------------------
# Optuna tuner (optional)
# ---------------------------
def _maybe_optuna_params(
    X_df: pd.DataFrame, y: np.ndarray, trials: int
) -> Dict[str, Any]:
    default = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "n_estimators": 4000,
        "num_leaves": 41,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "force_col_wise": True,
        "n_jobs": -1,
    }
    if trials <= 0:
        return default

    try:
        import optuna
    except ImportError:
        bt.logging.info("[optuna] Not installed; using default params.")
        return default

    bt.logging.info(f"[optuna] Starting study with {trials} trials...")

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 6000, step=250),
            "num_leaves": trial.suggest_int("num_leaves", 15, 128),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "force_col_wise": True,
            "n_jobs": -1,
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        maes = []
        for tr, va in kf.split(X_df):
            Xtr, Xva = X_df.iloc[tr], X_df.iloc[va]
            ytr, yva = y[tr], y[va]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
            )
            preds = model.predict(Xva)
            maes.append(mean_absolute_error(yva, preds))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)

    best = study.best_trial.params
    best.update(
        {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "force_col_wise": True,
            "n_jobs": -1,
        }
    )
    bt.logging.info("[optuna] Best params:\n" + json.dumps(best, indent=2))
    return best


def train_and_register(trials: int = 0):
    """
    trials=0  -> random search over N_PARAM_SAMPLES and OOF selection
    trials>0  -> Optuna to pick best params, then OOF with that config
    """
    ensure_indexes()
    X_raw, y = load_xy_full()
    if len(y) < 100:
        bt.logging.info(f"Not enough rows to train (have {len(y)}).")
        return

    X_raw, y = remove_value_outliers(X_raw, y, low=1, high=99)

    # Confidence-aware FE
    X_df, feat_names = fe_transform_df_from_raw(X_raw)

    if PRUNE_RESIDUAL_OUTLIERS:
        X_df, y = prune_residual_outliers(X_df, y, pct=2.0)

    # --- Tuning path selection ---
    if trials > 0:
        # Use Optuna to get best params
        best_params = _maybe_optuna_params(X_df, y, trials)
        cv_mae, cv_std, oof = cv_with_oof_and_params(X_df, y, best_params)
        bt.logging.info(
            f"[optuna] CV MAE={cv_mae:.3f} (±{cv_std:.3f}) with tuned params."
        )
        best = {"mae": cv_mae, "std": cv_std, "params": best_params, "oof": oof}
    else:
        # Random search (your original approach)
        best = {"mae": 1e9, "std": 0.0, "params": None, "oof": None}
        for i in range(N_PARAM_SAMPLES):
            params = sample_param()
            cv_mae, cv_std, oof = cv_with_oof_and_params(X_df, y, params)
            bt.logging.info(
                f"[{i+1:02d}/{N_PARAM_SAMPLES}] MAE={cv_mae:.3f} (±{cv_std:.3f}) params={params}"
            )
            if cv_mae < best["mae"]:
                best = {"mae": cv_mae, "std": cv_std, "params": params, "oof": oof}

    bt.logging.info(
        f"\nBest CV MAE={best['mae']:.3f} (±{best['std']:.3f}) with params:"
    )
    bt.logging.info(best["params"])

    # Isotonic calibration on OOF
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(best["oof"], y)

    # Calibrated holdout
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.15, random_state=RANDOM_STATE
    )
    final_params = best["params"].copy()

    hold = lgb.LGBMRegressor(**final_params)
    hold.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    raw_va = hold.predict(X_va, num_iteration=hold.best_iteration_)
    val_mae = float(mean_absolute_error(y_va, iso.predict(raw_va)))

    # Final model on full data (no early stopping)
    final = lgb.LGBMRegressor(**final_params)
    final.fit(X_df, y, callbacks=[lgb.log_evaluation(period=0)])

    # Serialize model + calibrator
    import joblib

    buf_model, buf_iso = io.BytesIO(), io.BytesIO()
    joblib.dump(final, buf_model)
    joblib.dump(iso, buf_iso)

    doc = {
        "createdAt": time.time(),
        "algo": "lightgbm.LGBMRegressor",
        "feature_order": feat_names,
        "base_feature_order": [f"X[{i}]" for i in range(20)],
        "metrics": {
            "cv_mae": float(best["mae"]),
            "cv_std": float(best["std"]),
            "val_mae": val_mae,
            "n_rows": int(len(y)),
            "n_splits": N_SPLITS,
            "tuning": "optuna" if trials > 0 else "random",
            "trials": int(trials),
        },
        "params": final_params,
        "joblib_blob": buf_model.getvalue(),
        "calibrator_blob": buf_iso.getvalue(),
    }
    models_col.insert_one(doc)
    bt.logging.info(
        f"[OK] Stored tuned model. CV_MAE={best['mae']:.3f}, VAL_MAE={val_mae:.3f}, rows={len(y)}"
    )


if __name__ == "__main__":
    bt.logging.set_trace()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    # Change trials as desired; 0 uses random search.
    train_and_register(trials=60)
