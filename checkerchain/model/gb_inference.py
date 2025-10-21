from __future__ import annotations
import io
from typing import Dict, Any, List
from joblib import load

from checkerchain.database.mongo import models_col
from checkerchain.model.fe import fe_transform_one_df

_cached = {"model": None, "feature_order": None, "createdAt": None}


def _load_latest():
    doc = models_col.find_one({}, sort=[("createdAt", -1)])
    if not doc:
        return None, None
    model = load(io.BytesIO(doc["joblib_blob"]))
    return model, doc


def predict_from_breakdown(breakdown: Dict[str, float]) -> float:
    if _cached["model"] is None:
        model, doc = _load_latest()
        if model is None:
            raise RuntimeError("No trained model available in cc_models.")
        _cached["model"] = model
        _cached["feature_order"] = list(
            doc.get("feature_order", [])
        )  # engineered order
        _cached["createdAt"] = doc.get("createdAt")

    # Build a 1-row DataFrame aligned to the saved feature order
    X_df = fe_transform_one_df(breakdown, feature_order=_cached["feature_order"])

    # Predict with named columns to avoid sklearn's feature-name warning
    yhat = float(_cached["model"].predict(X_df)[0])

    # Clamp to 0..100
    return max(0.0, min(100.0, yhat))
