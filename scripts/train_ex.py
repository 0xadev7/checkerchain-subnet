from __future__ import annotations

import numpy as np
import pandas as pd
import io, time, random

import warnings, json
from typing import List, Tuple, Dict, Any, Callable

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CORE_IDX = list(range(0, 10))  # X[0]..X[9]
CONF_IDX = list(range(10, 20))  # X[10]..X[19]
EPS = 1e-6


from checkerchain.database.mongo import (
    dataset_col,
    ensure_indexes,
)


def load_xy_full() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


# ----------------------------
# Feature Engineering
# ----------------------------
def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_features_from_raw_X(raw_X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build confidence-aware features from a raw X with columns X[0]..X[19].
    - X[0..9]   = core
    - X[10..19] = confidence (paired: (X[0],X[10]), (X[1],X[11]), ...)
    Returns (feats_df, feature_names)
    """
    expected_cols = [f"X[{i}]" for i in range(20)]
    missing = [c for c in expected_cols if c not in raw_X.columns]
    if missing:
        raise ValueError(f"X is missing required columns: {missing}")

    df = raw_X[expected_cols].copy()
    core_cols = [f"X[{i}]" for i in CORE_IDX]
    conf_cols = [f"X[{i}]" for i in CONF_IDX]

    df = _ensure_numeric(df, core_cols + conf_cols)

    # Gentle clipping for robustness (kept wide)
    df[core_cols] = df[core_cols].clip(lower=-5, upper=15)
    df[conf_cols] = df[conf_cols].clip(lower=0.0, upper=1.0)

    feats = pd.DataFrame(index=df.index)

    # 1) Per-dimension transforms
    for i in range(10):
        ccol = f"X[{i}]"
        fcol = f"X[{10+i}]"

        core = df[ccol]
        conf = df[fcol]
        conf2 = conf**2
        core2 = core**2

        feats[f"{ccol}_core"] = core
        feats[f"{fcol}_conf"] = conf
        feats[f"{fcol}_conf2"] = conf2

        feats[f"{ccol}_w1"] = core * conf  # core * conf
        feats[f"{ccol}_w2"] = core * conf2  # core * conf^2
        feats[f"{ccol}_quad_w"] = core2 * conf  # core^2 * conf

        feats[f"{ccol}_over_conf"] = core / (conf + EPS)  # core / conf
        feats[f"{ccol}_minus_10conf"] = core - 10.0 * conf

    # 2) Global aggregates
    core = df[core_cols]
    conf = df[conf_cols]

    feats["core_mean"] = core.mean(axis=1)
    feats["core_std"] = core.std(axis=1)
    feats["core_min"] = core.min(axis=1)
    feats["core_max"] = core.max(axis=1)
    feats["core_range"] = feats["core_max"] - feats["core_min"]
    feats["core_sum"] = core.sum(axis=1)

    feats["conf_mean"] = conf.mean(axis=1)
    feats["conf_std"] = conf.std(axis=1)
    feats["conf_sum"] = conf.sum(axis=1)

    weighted_core = core.values * conf.values
    feats["wcore_mean"] = weighted_core.mean(axis=1)
    feats["wcore_sum"] = weighted_core.sum(axis=1)
    feats["wcore_std"] = pd.DataFrame(weighted_core, index=df.index).std(axis=1)

    # 3) Pairwise weighted interactions (45 terms)
    for i in range(10):
        for j in range(i + 1, 10):
            feats[f"int_w_{i}_{j}"] = (
                df[f"X[{i}]"] * df[f"X[{j}]"] * df[f"X[{10+i}]"] * df[f"X[{10+j}]"]
            )

    # 4) Mild nonlinearity across first 5 pairs
    for i in range(5):
        feats[f"core2_conf2_{i}"] = (df[f"X[{i}]"] ** 2) * (df[f"X[{10+i}]"] ** 2)

    return feats, list(feats.columns)


# ----------------------------
# Modeling
# ----------------------------
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


def _maybe_optuna_params(X: np.ndarray, y: np.ndarray, n_trials: int) -> dict:
    if n_trials <= 0:
        return _default_lgbm_params()
    try:
        import optuna  # type: ignore
    except Exception:
        print("[INFO] Optuna not installed; using default LightGBM params.")
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
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
            )
            preds = model.predict(X_va)
            maes.append(mean_absolute_error(y_va, preds))
        return float(np.mean(maes))

    print(f"[INFO] Starting Optuna search for {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
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


def _cross_validate(
    X: np.ndarray, y: np.ndarray, params: dict
) -> Tuple[float, float, List[lgb.LGBMRegressor]]:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    maes, models = [], []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        preds = model.predict(X_va)
        mae = mean_absolute_error(y_va, preds)
        maes.append(mae)
        models.append(model)
        print(f"[FOLD {fold}] MAE = {mae:.4f} (best_iter={model.best_iteration_})")
    return float(np.mean(maes)), float(np.std(maes)), models


def _summarize_importance(
    models: List[lgb.LGBMRegressor], feature_names: List[str], top_k: int = 40
) -> pd.DataFrame:
    gain = np.zeros(len(feature_names), dtype=float)
    for m in models:
        gain += m.booster_.feature_importance(importance_type="gain")
    gain /= max(len(models), 1)
    order = np.argsort(-gain)
    top_idx = order[:top_k]
    imp_df = pd.DataFrame(
        {"feature": [feature_names[i] for i in top_idx], "gain": gain[top_idx]}
    )
    print("\nTop feature importances (avg gain):")
    for _, r in imp_df.iterrows():
        print(f"{r['feature']:40s} {r['gain']:10.3f}")
    return pd.DataFrame({"feature": feature_names, "gain": gain}).sort_values(
        "gain", ascending=False
    )


# ----------------------------
# Public API
# ----------------------------
def train_confidence_aware_regressor(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    trials: int = 0,
    holdout_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Train LightGBM on confidence-aware engineered features.
    - X: DataFrame with columns X[0]..X[19]
    - y: Series or single-column DataFrame with the target
    - trials: Optuna trials (0 to skip tuning)
    Returns dict with:
      - 'cv_mae_mean', 'cv_mae_std'
      - 'holdout_mae'
      - 'models' (list of CV models + final model)
      - 'final_model'
      - 'feature_names'
      - 'feature_importances_df'
      - 'predict_fn' (callable for raw X -> y_pred)
    """
    # y to 1D array
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly one column.")
        y_array = pd.to_numeric(y.iloc[:, 0], errors="coerce").values
    else:
        y_array = pd.to_numeric(y, errors="coerce").values

    # Feature engineering
    feats, feat_names = build_features_from_raw_X(X)

    mask = np.isfinite(y_array) & np.isfinite(feats.values).all(axis=1)
    X_fe = feats.values[mask]
    y_fe = y_array[mask]

    print(f"[INFO] FE dataset shape: N={X_fe.shape[0]}  d={X_fe.shape[1]}")

    # Params (Optuna optional)
    params = _maybe_optuna_params(X_fe, y_fe, n_trials=trials)

    # CV
    cv_mean, cv_std, cv_models = _cross_validate(X_fe, y_fe, params)
    print(f"\n[CV] MAE = {cv_mean:.4f} ± {cv_std:.4f}")

    # Final fit with small holdout for early stopping
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_fe, y_fe, test_size=holdout_ratio, random_state=RANDOM_STATE
    )
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
    )
    holdout_preds = final_model.predict(X_va)
    holdout_mae = mean_absolute_error(y_va, holdout_preds)
    print(
        f"[HOLDOUT] MAE = {holdout_mae:.4f} (best_iter={final_model.best_iteration_})"
    )

    # Importances
    all_models = cv_models + [final_model]
    imp_df = _summarize_importance(all_models, feat_names, top_k=40)

    # Predict function that applies the SAME FE to raw X
    def predict_fn(raw_X: pd.DataFrame) -> np.ndarray:
        feats_new, _ = build_features_from_raw_X(raw_X)
        # Align columns (in case of categorical drift or future edits)
        feats_new = feats_new.reindex(columns=feat_names, fill_value=0.0)
        return final_model.predict(feats_new.values)

    return {
        "cv_mae_mean": cv_mean,
        "cv_mae_std": cv_std,
        "holdout_mae": holdout_mae,
        "models": all_models,
        "final_model": final_model,
        "feature_names": feat_names,
        "feature_importances_df": imp_df,
        "predict_fn": predict_fn,
    }


# ----------------------------
# Minimal convenience: run if X, y exist in globals
# (Safe no-op otherwise)
# ----------------------------
if __name__ == "__main__":
    X, y = load_xy_full()

    out = train_confidence_aware_regressor(X, y, trials=0)
    print("\nSummary:")
    print(f"CV MAE: {out['cv_mae_mean']:.4f} ± {out['cv_mae_std']:.4f}")
    print(f"Holdout MAE: {out['holdout_mae']:.4f}")
