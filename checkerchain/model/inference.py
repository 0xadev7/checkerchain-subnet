from __future__ import annotations
import io
from typing import Dict
from joblib import load

from checkerchain.database.mongo import models_col
from checkerchain.model.fe import fe_transform_one_df

_cached = {"model": None, "iso": None, "feature_order": None, "createdAt": None}


def _load_latest():
    doc = models_col.find_one({}, sort=[("createdAt", -1)])
    if not doc:
        return None, None
    model = load(io.BytesIO(doc["joblib_blob"]))
    iso = None
    if "calibrator_blob" in doc:
        iso = load(io.BytesIO(doc["calibrator_blob"]))
    return (model, iso), doc


def predict_from_breakdown_and_confidence(breakdown: Dict[str, float], confidence: Dict[str, float]) -> float:
    if _cached["model"] is None:
        pair, doc = _load_latest()
        if pair is None:
            raise RuntimeError("No trained model available in cc_models.")
        _cached["model"], _cached["iso"] = pair
        _cached["feature_order"] = list(doc.get("feature_order", []))
        _cached["createdAt"] = doc.get("createdAt")

    X_df = fe_transform_one_df(breakdown, feature_order=_cached["feature_order"])
    raw = float(_cached["model"].predict(X_df)[0])
    if _cached["iso"] is not None:
        raw = float(_cached["iso"].predict([raw])[0])
    return max(0.0, min(100.0, raw))
