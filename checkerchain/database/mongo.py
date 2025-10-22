from __future__ import annotations
import os, time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "checkerchain")

client = MongoClient(MONGO_URI, retryWrites=True)
db = client[MONGO_DB]

# Collections
raw_col: Collection = db["cc_products_raw"]
assess_col: Collection = db["cc_assessments"]
labels_col: Collection = db["cc_labels"]
dataset_col: Collection = db["cc_datasets"]
models_col: Collection = db["cc_models"]
meta_col: Collection = db["cc_meta"]

METRICS = [
    "project",
    "utility",
    "userbase",
    "team",
    "security",
    "audit_verification",
    "tokenomics",
    "sustainability",
    "decentralization",
    "governance_participation",
    "marketing",
    "community_engagement",
    "partnerships",
    "developer_ecosystem",
    "roadmap",
    "clarity",
    "reputation",
]


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def upsert_meta(key: str, value: Any):
    meta_col.update_one(
        {"key": key}, {"$set": {"value": value, "ts": utcnow_iso()}}, upsert=True
    )


def get_meta(key: str, default=None):
    doc = meta_col.find_one({"key": key})
    return doc["value"] if doc and "value" in doc else default


# Fast indices (run once, safe to repeat)
def ensure_indexes():
    raw_col.create_index(
        [("id", ASCENDING), ("currentReviewCycle", ASCENDING)], unique=False
    )
    assess_col.create_index(
        [("productId", ASCENDING), ("reviewCycle", ASCENDING)], unique=True
    )
    labels_col.create_index(
        [("productId", ASCENDING), ("reviewCycle", ASCENDING)], unique=True
    )
    dataset_col.create_index(
        [("productId", ASCENDING), ("reviewCycle", ASCENDING)], unique=True
    )
    models_col.create_index([("createdAt", DESCENDING)])


def save_breakdown(
    product_id: str,
    review_cycle: int,
    x: list[float],
    model_version: str = "gb_v1",
) -> None:
    """
    Save or update the 10-metric breakdown for a product/review cycle.
    This is called by llm.generate_complete_assessment(...) after LLM scoring.

    Args:
        product_id: CheckerChain product ID
        review_cycle: current review cycle integer
        x: list of 10 floats (metrics 0–10)
        model_version: version tag for tracking inference model
    """
    if len(x) != len(METRICS):
        raise ValueError(f"Expected {len(METRICS)} metrics, got {len(x)}")

    doc = {
        "productId": product_id,
        "reviewCycle": int(review_cycle),
        "breakdown": {k: float(v) for k, v in zip(METRICS, x)},
        "modelVersion": model_version,
        "ts": time.time(),
    }

    assess_col.update_one(
        {"productId": product_id, "reviewCycle": int(review_cycle)},
        {"$set": doc},
        upsert=True,
    )

    # If label already exists, also upsert into cc_datasets for training convenience
    lbl = labels_col.find_one(
        {"productId": product_id, "reviewCycle": int(review_cycle)}
    )
    if lbl and "y" in lbl:
        dataset_col.update_one(
            {"productId": product_id, "reviewCycle": int(review_cycle)},
            {
                "$set": {
                    "X": x,
                    "y": float(lbl["y"]),
                    "ts": time.time(),
                    "modelVersion": model_version,
                }
            },
            upsert=True,
        )
