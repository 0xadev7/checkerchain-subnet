from __future__ import annotations
import io, time, json
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump

from checkerchain.database.mongo import (
    dataset_col,
    models_col,
    METRICS,
    ensure_indexes,
    upsert_meta,
)


def load_xy() -> Tuple[np.ndarray, np.ndarray]:
    rows = list(dataset_col.find({}, {"X": 1, "y": 1}))
    X = np.array([r["X"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    y = np.array([r["y"] for r in rows if ("X" in r and "y" in r)], dtype=float)
    return X, y


def train_and_register():
    ensure_indexes()
    X, y = load_xy()
    if len(X) < 100:
        print(f"Not enough rows to train (have {len(X)}).")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Strong defaults for tabular regression
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        max_iter=600,
        learning_rate=0.05,
        l2_regularization=1e-6,
        early_stopping=True,  # uses a validation split internally; still okay with our X_val to evaluate
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    pred_val = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))
    r2 = float(r2_score(y_val, pred_val))
    print(f"Validation: MAE={mae:.3f}, R2={r2:.4f}, N={len(y_val)}")

    # Serialize model to bytes (joblib)
    buf = io.BytesIO()
    dump(model, buf)
    model_blob = buf.getvalue()

    # Save to registry
    doc = {
        "createdAt": time.time(),
        "algo": "sklearn.HistGradientBoostingRegressor",
        "feature_order": METRICS,
        "metrics": {"val_mae": mae, "val_r2": r2, "n": len(y)},
        "joblib_blob": model_blob,
    }
    models_col.insert_one(doc)
    upsert_meta("latest_model_createdAt", doc["createdAt"])
    print("[OK] Model stored in cc_models")


if __name__ == "__main__":
    train_and_register()
