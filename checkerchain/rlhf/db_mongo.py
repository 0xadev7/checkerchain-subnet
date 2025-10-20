from __future__ import annotations
import os, time
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING, ReturnDocument
from pymongo.collection import Collection
import numpy as np

from checkerchain.rlhf.constants import METRICS, N_METRICS, DEFAULT_W


class RLHFMongo:
    """
    Collections:
      - miner_breakdowns: one row per (product_id, review_cycle)
      - miner_targets:    one row per (product_id, review_cycle) once trustScore is seen
      - miner_weights:    versioned weights (latest picked by created_at desc)
      - rlhf_meta:        single doc for cursors (e.g., last_products_cursor_ts)
    """

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = db_name or os.getenv("MONGO_DB", "checkerchain")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.col_breakdowns: Collection = self.db["miner_breakdowns"]
        self.col_targets: Collection = self.db["miner_targets"]
        self.col_weights: Collection = self.db["miner_weights"]
        self.col_meta: Collection = self.db["rlhf_meta"]

        self._ensure_indexes()

    def _ensure_indexes(self):
        self.col_breakdowns.create_index(
            [("product_id", ASCENDING), ("review_cycle", ASCENDING)], unique=True
        )
        self.col_targets.create_index(
            [("product_id", ASCENDING), ("review_cycle", ASCENDING)], unique=True
        )
        self.col_weights.create_index([("created_at", ASCENDING)])

    # ---------------- save/load ----------------

    def save_breakdown(
        self,
        *,
        product_id: str,
        review_cycle: int,
        x: List[float],
        model_version: str,
        created_at: Optional[float] = None,
    ):
        assert len(x) == N_METRICS
        ts = created_at or time.time()
        doc = {
            "product_id": product_id,
            "review_cycle": int(review_cycle),
            "x": [float(v) for v in x],
            "metrics": METRICS,
            "model_version": model_version,
            "created_at": ts,
        }
        self.col_breakdowns.update_one(
            {"product_id": product_id, "review_cycle": int(review_cycle)},
            {"$set": doc},
            upsert=True,
        )

    def save_target_if_new(
        self,
        *,
        product_id: str,
        review_cycle: int,
        trust_score: float,
        published_at: float,
    ) -> bool:
        res = self.col_targets.update_one(
            {"product_id": product_id, "review_cycle": int(review_cycle)},
            {
                "$setOnInsert": {
                    "product_id": product_id,
                    "review_cycle": int(review_cycle),
                    "trust_score": float(trust_score),
                    "published_at": float(published_at),
                    "created_at": time.time(),
                }
            },
            upsert=True,
        )
        # matched_count==0 implies insert; modified_count may be 0 on duplicate
        # Use find to be sure:
        existed = self.col_targets.find_one(
            {"product_id": product_id, "review_cycle": int(review_cycle)}
        )
        return (
            existed is not None
            and abs(existed.get("trust_score", -1) - trust_score) < 1e-6
        )

    def load_weights(self) -> List[float]:
        doc = self.col_weights.find_one(sort=[("created_at", -1)])
        if not doc:
            return list(DEFAULT_W)
        w = doc.get("w", DEFAULT_W)
        if len(w) != N_METRICS:
            return list(DEFAULT_W)
        s = sum(w)
        if s <= 0:
            return list(DEFAULT_W)
        return [float(v) / s for v in w]

    def save_weights(self, w: List[float], meta: Dict[str, Any] | None = None):
        doc = {
            "w": [float(v) for v in w],
            "created_at": time.time(),
            "meta": meta or {},
        }
        self.col_weights.insert_one(doc)

    def get_all_breakdowns_with_targets(self) -> List[Tuple[list, float, float]]:
        """
        Returns list of (x10, y100, published_ts)
        for all pairs that exist in both collections (latest cycle per product).
        """
        rows = []
        # join in python for simplicity (tiny collections)
        tgt_map = {}
        for t in self.col_targets.find({}):
            key = (t["product_id"], t["review_cycle"])
            tgt_map[key] = (float(t["trust_score"]), float(t["published_at"]))
        for b in self.col_breakdowns.find({}):
            key = (b["product_id"], b["review_cycle"])
            if key in tgt_map:
                y, ts = tgt_map[key]
                rows.append((b["x"], y, ts))
        return rows

    def get_new_pairs_since(
        self, since_ts: float
    ) -> List[Tuple[str, list, float, float]]:
        """
        Returns list of (product_id, x10, y100, published_ts) where target doc is newer than since_ts.
        """
        rows = []
        tgt_map = {}
        for t in self.col_targets.find({"created_at": {"$gt": float(since_ts)}}):
            tgt_map[(t["product_id"], t["review_cycle"])] = (
                float(t["trust_score"]),
                float(t["published_at"]),
            )
        if not tgt_map:
            return rows
        for b in self.col_breakdowns.find({}):
            key = (b["product_id"], b["review_cycle"])
            if key in tgt_map:
                y, ts = tgt_map[key]
                rows.append((b["product_id"], b["x"], y, ts))
        return rows

    # --------------- meta cursor ---------------

    def get_last_update_ts(self) -> float:
        doc = self.col_meta.find_one({"_id": "rlhf_last_update_ts"})
        return float(doc["ts"]) if doc and "ts" in doc else 0.0

    def set_last_update_ts(self, ts: float):
        self.col_meta.update_one(
            {"_id": "rlhf_last_update_ts"},
            {"$set": {"ts": float(ts)}},
            upsert=True,
        )
