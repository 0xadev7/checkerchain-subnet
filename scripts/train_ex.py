# train_confidence_aware_from_mongo.py
from __future__ import annotations
import warnings, json
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ----------------------------
# Mongo Import
# ----------------------------
from checkerchain.database.mongo import dataset_col  # adjust import path if needed

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CORE_IDX = list(range(0, 10))
CONF_IDX = list(range(10, 20))
EPS = 1e-6


# ======================================================
# === 1. Mongo Loader
# ======================================================
def load_xy_full() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


# ======================================================
# === 2. Feature Engineering
# ======================================================
def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_features_from_raw_X(raw_X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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


# ======================================================
# === 3. Modeling Helpers
# ======================================================
def _default_lgbm_params() -> dict:
    return {
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
    }


def _maybe_optuna_params(X: np.ndarray, y: np.ndarray, trials: int) -> dict:
    if trials <= 0:
        return _default_lgbm_params()
    try:
        import optuna
    except ImportError:
        print("[INFO] Optuna not installed; using defaults.")
        return _default_lgbm_params()

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
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        maes = []
        for tr, va in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
            )
            preds = model.predict(X[va])
            maes.append(mean_absolute_error(y[va], preds))
        return float(np.mean(maes))

    print(f"[INFO] Running Optuna ({trials} trials)...")
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
        }
    )
    print("[INFO] Best Optuna params:", json.dumps(best, indent=2))
    return best


def _cross_validate(X: np.ndarray, y: np.ndarray, params: dict):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    maes, models = [], []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[tr],
            y[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        preds = model.predict(X[va])
        mae = mean_absolute_error(y[va], preds)
        maes.append(mae)
        models.append(model)
        print(f"[FOLD {fold}] MAE={mae:.4f}")
    return float(np.mean(maes)), float(np.std(maes)), models


def _summarize_importance(models, feat_names):
    gain = np.zeros(len(feat_names))
    for m in models:
        gain += m.booster_.feature_importance(importance_type="gain")
    gain /= max(len(models), 1)
    order = np.argsort(-gain)
    print("\nTop 30 feature importances:")
    for i in order[:30]:
        print(f"{feat_names[i]:40s} {gain[i]:10.3f}")


# ======================================================
# === 4. Main Trainer
# ======================================================
def train_from_mongo(trials: int = 0, holdout_ratio: float = 0.15):
    print("[INFO] Loading data from Mongo...")
    X_arr, y_arr = load_xy_full()
    print(f"[INFO] Loaded N={len(y_arr)}, D={X_arr.shape[1]}")

    # Convert NumPy → DataFrame
    X_df = pd.DataFrame(X_arr, columns=[f"X[{i}]" for i in range(X_arr.shape[1])])
    y_series = pd.Series(y_arr, name="y")

    # Build features
    feats, feat_names = build_features_from_raw_X(X_df)
    mask = np.isfinite(y_series) & np.isfinite(feats.values).all(axis=1)
    Xf, yf = feats.values[mask], y_series.values[mask]
    print(f"[INFO] After FE: N={Xf.shape[0]}, d={Xf.shape[1]}")

    # Params
    params = _maybe_optuna_params(Xf, yf, trials)
    cv_mean, cv_std, models = _cross_validate(Xf, yf, params)
    print(f"\n[CV] MAE={cv_mean:.4f} ± {cv_std:.4f}")

    # Final fit
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xf, yf, test_size=holdout_ratio, random_state=RANDOM_STATE
    )
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
    )
    preds = final_model.predict(X_va)
    holdout_mae = mean_absolute_error(y_va, preds)
    print(f"[HOLDOUT] MAE={holdout_mae:.4f}")

    _summarize_importance(models + [final_model], feat_names)
    return {
        "cv_mae_mean": cv_mean,
        "cv_mae_std": cv_std,
        "holdout_mae": holdout_mae,
        "final_model": final_model,
        "feature_names": feat_names,
    }


# ======================================================
# === 5. Auto-run if main
# ======================================================
if __name__ == "__main__":
    results = train_from_mongo(trials=60)  # set trials=0 to skip Optuna
    print("\n✅ Training complete:")
    print(f"CV MAE = {results['cv_mae_mean']:.4f} ± {results['cv_mae_std']:.4f}")
    print(f"Holdout MAE = {results['holdout_mae']:.4f}")
