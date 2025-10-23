from __future__ import annotations
import os, time
from datetime import datetime, timezone, timedelta
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
webcache_col: Collection = db["cc_webcache"]

METRICS = [
    "project",
    "utility",
    "userbase",
    "team",
    "security",
    "tokenomics",
    "marketing",
    "partnerships",
    "roadmap",
    "clarity",
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

    webcache_col.create_index(
        [("productId", ASCENDING), ("reviewCycle", ASCENDING)],
        unique=True,
        name="prod_cycle_unique",
    )
    ttl_days = int(os.getenv("WEB_URL_CACHE_TTL_DAYS", "7"))
    try:
        webcache_col.create_index(
            [("createdAt", ASCENDING)],
            expireAfterSeconds=ttl_days * 24 * 3600,
            name="createdAt_ttl",
        )
    except Exception:
        # Some Mongo versions can't modify TTL via create_index; ignore if exists.
        pass


def save_breakdown_and_confidence(
    product_id: str,
    review_cycle: int,
    x: list[float],
    cx: list[float],
    model_version: str = "gb_v1",
) -> None:
    """
    Save or update the 10-metric breakdown for a product/review cycle.
    This is called by llm.generate_complete_assessment(...) after LLM scoring.
    """
    if len(x) != len(METRICS):
        raise ValueError(f"Expected {len(METRICS)} metrics, got {len(x)}")

    doc = {
        "productId": product_id,
        "reviewCycle": int(review_cycle),
        "breakdown": {k: float(v) for k, v in zip(METRICS, x)},
        "confidence": {k: float(v) for k, v in zip(METRICS, cx)},
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
                    "X": x + cx,
                    "y": float(lbl["y"]),
                    "ts": time.time(),
                    "modelVersion": model_version,
                }
            },
            upsert=True,
        )


def get_cached_urls(
    product_id: str,
    review_cycle: int,
) -> Optional[List[str]]:
    """
    Return cached URLs if present (and fresh enough when max_age_hours is given).
    """
    doc = webcache_col.find_one(
        {"productId": product_id, "reviewCycle": int(review_cycle)},
        {"urls": 1},
    )
    if not doc or "urls" not in doc:
        return None
    return list(doc["urls"])


def upsert_cached_urls(
    product_id: str,
    review_cycle: int,
    urls: List[str],
    source: str = "tavily",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.now(timezone.utc)
    webcache_col.update_one(
        {"productId": product_id, "reviewCycle": int(review_cycle)},
        {
            "$set": {
                "productId": product_id,
                "reviewCycle": int(review_cycle),
                "urls": list(urls),
                "source": source,
                "meta": meta or {},
                "updatedAt": now,
            },
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )


def invalidate_cached_urls(product_id: str, review_cycle: int) -> None:
    webcache_col.delete_one({"productId": product_id, "reviewCycle": int(review_cycle)})
