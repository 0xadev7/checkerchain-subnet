from __future__ import annotations
import io
from typing import Dict, Any, List
from joblib import load
from checkerchain.database.mongo import models_col

from checkerchain.model.fe import fe_transform_one

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
        _cached["feature_order"] = doc.get("feature_order")  # engineered order
        _cached["createdAt"] = doc.get("createdAt")

    # Build engineered feature vector
    x_vec, names = fe_transform_one(breakdown)

    # (Optional) assert name order match
    if _cached["feature_order"] and list(_cached["feature_order"]) != list(names):
        # If mismatch, reorder (defensive)
        name_to_idx = {n: i for i, n in enumerate(names)}
        ordered = [_cached["feature_order"], name_to_idx]  # avoid flake8 warning
        x_ordered = [x_vec[name_to_idx[n]] for n in _cached["feature_order"]]
        yhat = float(_cached["model"].predict([x_ordered])[0])
    else:
        yhat = float(_cached["model"].predict([x_vec])[0])

    return max(0.0, min(100.0, yhat))
