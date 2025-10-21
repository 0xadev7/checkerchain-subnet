from __future__ import annotations
import io
from typing import Dict, Any, List
from joblib import load

from checkerchain.database.mongo import models_col, METRICS

_cached = {"model": None, "feature_order": METRICS, "createdAt": None}


def _load_latest():
    doc = models_col.find_one({}, sort=[("createdAt", -1)])
    if not doc:
        return None, None
    blob = doc["joblib_blob"]
    model = load(io.BytesIO(blob))
    return model, doc


def predict_from_breakdown(breakdown: Dict[str, float]) -> float:
    # Lazy cache
    if _cached["model"] is None:
        model, doc = _load_latest()
        if model is None:
            # No model yet
            raise RuntimeError("No trained model available in cc_models.")
        _cached["model"] = model
        _cached["feature_order"] = doc.get("feature_order", METRICS)
        _cached["createdAt"] = doc.get("createdAt")

    X = [[float(breakdown.get(k, 0.0)) for k in _cached["feature_order"]]]
    yhat = float(_cached["model"].predict(X)[0])
    return max(0.0, min(100.0, yhat))  # clamp to [0,100]
