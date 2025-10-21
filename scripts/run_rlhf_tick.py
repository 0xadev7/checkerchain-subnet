from __future__ import annotations
import os, json, time, math, requests
import numpy as np
from typing import Dict, Any, List

from checkerchain.rlhf.db_mongo import RLHFMongo
from checkerchain.rlhf.job import fetch_recent_products  # reuse small pager
from checkerchain.rlhf.online import online_update
from checkerchain.rlhf.batch_fit import fit_weights_bias_scale
from checkerchain.rlhf.constants import DEFAULT_W


def _now_ts() -> float:
    return time.time()


def _iso_to_epoch(s: str | None) -> float:
    if not s:
        return _now_ts()
    try:
        return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return _now_ts()


def _ingest_targets_from_api(db: RLHFMongo, pages: int) -> int:
    products = fetch_recent_products(limit_pages=pages)
    inserts = 0
    for p in products:
        if p.get("status") != "reviewed":
            continue
        if "trustScore" not in p:
            continue
        product_id = str(p.get("_id") or p.get("id"))
        rc = int(p.get("currentReviewCycle", 1))
        y = float(p.get("trustScore", 0.0))
        ts = _iso_to_epoch(
            p.get("lastReviewed") or p.get("updatedAt") or p.get("createdAt")
        )
        # upsert once; if already exists, this is a no-op
        db.save_target_if_new(
            product_id=product_id, review_cycle=rc, trust_score=y, published_at=ts
        )
        inserts += 1
    return inserts


def main():
    # -------- thresholds & knobs (env-configurable) --------
    FETCH_PAGES = int(os.getenv("RLHF_FETCH_PAGES", "5"))
    LR = float(os.getenv("RLHF_LR", "0.05"))
    LAMBDA_STABILITY = float(os.getenv("RLHF_LAMBDA", "0.01"))
    HUBER_DELTA_ONLINE = float(os.getenv("RLHF_HUBER_DELTA", "0.75"))
    HALF_LIFE_DAYS = int(os.getenv("RLHF_HALF_LIFE_DAYS", "14"))
    PGD_STEPS = int(os.getenv("RLHF_PGD_STEPS", "3"))

    # Auto-refit triggers
    REFIT_MIN_NEW = int(
        os.getenv("RLHF_REFIT_MIN_NEW", "150")
    )  # refit if >= this many new labeled pairs since last batch
    REFIT_MAX_AGE_HOURS = int(
        os.getenv("RLHF_REFIT_MAX_AGE_HOURS", "168")
    )  # refit if last batch fit older than this (default 7 days)

    # Batch-fit knobs
    HUBER_DELTA_BATCH = float(os.getenv("RLHF_BATCH_HUBER_DELTA", "3.0"))
    LAMBDA_W_L2 = float(os.getenv("RLHF_BATCH_L2", "1e-3"))
    LAMBDA_W_TV = float(os.getenv("RLHF_BATCH_TV", "0.0"))

    db = RLHFMongo()

    # 1) Ingest any newly published targets
    _ = _ingest_targets_from_api(db, FETCH_PAGES)

    # 2) Determine whether we should trigger a full batch refit
    last_batch_ts = db.get_last_batch_fit_ts()
    now = _now_ts()
    hours_since_batch = (now - last_batch_ts) / 3600.0

    # Count new supervised pairs since last batch
    new_pairs = db.get_new_pairs_since(last_batch_ts)  # [(product_id, x10, y100, ts)]
    need_refit = (len(new_pairs) >= REFIT_MIN_NEW) or (
        hours_since_batch >= REFIT_MAX_AGE_HOURS
    )

    if need_refit:
        # ----- Full batch fit on ALL pairs -----
        rows = db.get_all_breakdowns_with_targets()  # [(x10, y100, ts)]
        if len(rows) >= 20:
            X = np.stack([r[0] for r in rows])
            y = np.array([r[1] for r in rows])
            w_hat, b0, b1, obj = fit_weights_bias_scale(
                X,
                y,
                huber_delta=HUBER_DELTA_BATCH,
                lambda_u_l2=LAMBDA_W_L2,
                lambda_u_tv=LAMBDA_W_TV,
            )
            db.save_weights(
                list(w_hat),
                {
                    "reason": "batch_fit",
                    "b0": b0,
                    "b1": b1,
                    "obj": obj,
                    "n": int(len(rows)),
                },
            )
            db.set_last_batch_fit_ts(now)
            print(
                json.dumps(
                    {
                        "updated": True,
                        "mode": "batch_fit",
                        "n": len(rows),
                        "b0": b0,
                        "b1": b1,
                        "obj": obj,
                    }
                )
            )
            return
        # not enough rows → fall through to online update

    # 3) Otherwise, do an ONLINE update on the new pairs (if any)
    if not new_pairs:
        print(json.dumps({"updated": False, "reason": "no-new-targets"}))
        return

    Xb = np.stack([row[1] for row in new_pairs])  # x10
    Yb = np.array([row[2] for row in new_pairs])  # y100
    Ts = [row[3] for row in new_pairs]

    # recency weights
    if HALF_LIFE_DAYS > 0:
        hl = HALF_LIFE_DAYS * 86400.0
        rec = np.array([0.5 ** ((now - ts) / hl) for ts in Ts], dtype=float)
    else:
        rec = np.ones(len(Ts), dtype=float)

    # Load current weights AND meta (carry b0,b1 forward)
    w_prev, meta_prev = db.load_weights_with_meta()
    b0 = float(meta_prev.get("b0", 0.0))
    b1 = float(meta_prev.get("b1", 1.0))

    w_new = online_update(
        np.array(w_prev, dtype=float),
        Xb,
        Yb,
        lr=LR,
        lambda_stability=LAMBDA_STABILITY,
        huber_delta=HUBER_DELTA_ONLINE,
        recency_weights=rec,
        pgd_steps=PGD_STEPS,
    )

    # Save with SAME b0, b1 (preserve calibration) – they’re only updated by batch_fit
    db.save_weights(
        list(w_new),
        {
            "reason": "online_update",
            "batch": int(len(new_pairs)),
            "huber_delta": HUBER_DELTA_ONLINE,
            "lambda": LAMBDA_STABILITY,
            "lr": LR,
            "half_life_days": HALF_LIFE_DAYS,
            "pgd_steps": PGD_STEPS,
            "b0": b0,
            "b1": b1,
            "since_batch_hours": hours_since_batch,
            "new_pairs": int(len(new_pairs)),
        },
    )
    print(
        json.dumps(
            {
                "updated": True,
                "mode": "online",
                "batch": len(new_pairs),
                "b0": b0,
                "b1": b1,
            }
        )
    )


if __name__ == "__main__":
    main()
