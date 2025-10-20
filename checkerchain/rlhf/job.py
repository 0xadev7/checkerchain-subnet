from __future__ import annotations
import os, time, math, requests
import numpy as np
from typing import Dict, Any, List

import bittensor as bt

from checkerchain.rlhf.online import online_update
from checkerchain.rlhf.init import initialize_weights_nnls
from checkerchain.rlhf.constants import N_METRICS, DEFAULT_W
from checkerchain.rlhf.db_mongo import RLHFMongo

API_URL = os.getenv("CHECKERCHAIN_API", "https://api.checkerchain.com/api/v1/products")
PAGE_SIZE = int(os.getenv("RLHF_PAGE_SIZE", "30"))


def _now_ts() -> float:
    return time.time()


def _page(url: str, page: int, limit: int) -> list[dict]:
    r = requests.get(url, params={"page": page, "limit": limit}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["data"]["products"]


def fetch_recent_products() -> List[dict]:
    out: List[dict] = []
    p = 1
    while True:
        items = _page(API_URL, page=p, limit=PAGE_SIZE)
        if not items:
            break
        out.extend(items)
        if len(items) < PAGE_SIZE:
            break
        p += 1
    return out


def exponential_recency_weights(
    timestamps: list[float], half_life_days=14
) -> np.ndarray:
    if not timestamps:
        return np.array([], dtype=float)
    now = _now_ts()
    hl = half_life_days * 86400.0
    return np.array([0.5 ** ((now - ts) / hl) for ts in timestamps], dtype=float)


def run_training_tick(
    db: RLHFMongo,
    *,
    lr: float = 0.05,
    lambda_stability: float = 0.01,
    huber_delta: float = 0.75,
    half_life_days: int = 14,
    pgd_steps: int = 3,
) -> dict:
    """
    1) Fetch recent products (few pages).
    2) For each product with trustScore, upsert a miner_targets row keyed by (product_id, review_cycle).
    3) Gather newly inserted pairs since last tick; run PGD to update weights.
    4) On first run (no weights), do NNLS init over all pairs.
    """
    # Step 1–2: ingest targets from API
    products = fetch_recent_products()
    bt.logging.info(f"RLHF Training: Fetched {len(products)} products.")

    new_target_ts_max = 0.0
    for p in products:
        if p.get("status") != "reviewed":
            continue
        if "trustScore" not in p:
            continue
        product_id = str(p.get("_id") or p.get("id"))
        review_cycle = int(p.get("currentReviewCycle", 1))
        trust_score = float(p.get("trustScore", 0.0))
        published_at = p.get("lastReviewed") or p.get("updatedAt") or p.get("createdAt")
        # convert ISO to epoch
        try:
            published_ts = time.mktime(
                time.strptime(published_at[:19], "%Y-%m-%dT%H:%M:%S")
            )
        except Exception:
            published_ts = _now_ts()

        db.save_target_if_new(
            product_id=product_id,
            review_cycle=review_cycle,
            trust_score=trust_score,
            published_at=published_ts,
        )
        new_target_ts_max = max(new_target_ts_max, published_ts)

    # Step 3: collect new pairs since last tick
    last_ts = db.get_last_update_ts()
    new_pairs = db.get_new_pairs_since(last_ts)

    bt.logging.info(f"RLHF Training: Found {len(new_pairs)} new products")
    if not new_pairs:
        return {"updated": False, "reason": "no-new-targets"}

    Xb = np.stack([row[1] for row in new_pairs])  # x10
    Yb = np.array([row[2] for row in new_pairs])  # y100
    Ts = [row[3] for row in new_pairs]
    rec = exponential_recency_weights(Ts, half_life_days)

    # Step 4: load or init weights
    w_prev = np.array(db.load_weights(), dtype=float)
    if (np.asarray(w_prev) == np.asarray(DEFAULT_W)).all():
        # Try batch init if enough data exists
        all_rows = db.get_all_breakdowns_with_targets()
        if len(all_rows) >= 20:
            X_all = np.stack([r[0] for r in all_rows])
            Y_all = np.array([r[1] for r in all_rows])
            w0 = initialize_weights_nnls(X_all, Y_all)
            db.save_weights(list(w0), {"reason": "nnls_init", "count": len(all_rows)})
            w_prev = w0

    # Step 5: online PGD
    w_new = online_update(
        w_prev,
        Xb,
        Yb,
        lr=lr,
        lambda_stability=lambda_stability,
        huber_delta=huber_delta,
        recency_weights=rec,
        pgd_steps=pgd_steps,
    )
    db.save_weights(
        list(w_new),
        {
            "reason": "online_update",
            "batch": int(len(new_pairs)),
            "huber_delta": huber_delta,
            "lambda": lambda_stability,
            "lr": lr,
            "half_life_days": half_life_days,
            "pgd_steps": pgd_steps,
        },
    )

    # Move cursor
    db.set_last_update_ts(max(last_ts, new_target_ts_max or _now_ts()))
    return {"updated": True, "batch": int(len(new_pairs))}
