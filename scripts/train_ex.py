from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import io, time, random

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from checkerchain.database.mongo import (
    dataset_col,
    ensure_indexes,
)


def load_xy_full() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


def train_and_register():
    ensure_indexes()
    X, y = load_xy_full()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=100,
        verbose=False,
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f}")


if __name__ == "__main__":
    train_and_register()
