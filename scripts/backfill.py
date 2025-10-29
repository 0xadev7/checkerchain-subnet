from __future__ import annotations
import os, math, asyncio, time, json
from typing import Dict, Any, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import bittensor as bt

from checkerchain.database.mongo import (
    raw_col,
    labels_col,
    assess_col,
    dataset_col,
    ensure_indexes,
    METRICS,
    upsert_meta,
    get_meta,
)
from checkerchain.miner.llm import generate_complete_assessment

load_dotenv()
BASE = os.getenv("CHECKERCHAIN_API", "https://api.checkerchain.com/api/v1")
PAGE_SIZE = int(os.getenv("CC_PAGE_LIMIT", "30"))


def _rc(p: Dict[str, Any]) -> int:
    try:
        return int(p.get("currentReviewCycle") or 1)
    except Exception:
        return 1


def _extract_x_cx_from_assess(
    doc: Dict[str, Any],
) -> tuple[list[float], list[float], str]:
    """
    Reconstruct (x, cx, model_version) from an assessment document exactly like
    save_breakdown_and_confidence writes it:
      breakdown: {metric_name: float}
      confidence: {metric_name: float}
    Order is METRICS.
    If confidence is missing/partial, pad zeros to len(METRICS).
    """
    if not doc:
        return [], [], "unknown"

    bd = doc.get("breakdown") or {}
    cf = doc.get("confidence") or {}
    model_version = str(doc.get("modelVersion") or "unknown")

    x = [float(bd.get(m, 0.0)) for m in METRICS]
    cx = [float(cf.get(m, 0.0)) for m in METRICS]
    return x, cx, model_version


def _try_sync_dataset(p: Dict[str, Any]) -> None:
    """
    If label & assessment exist but cc_datasets is missing, create it
    with the same shape as the live upsert: X = x + cx, plus y, ts, modelVersion.
    """
    try:
        product_id = p["id"]
        rc = _rc(p)

        # Already synced?
        if dataset_col.find_one({"productId": product_id, "reviewCycle": rc}):
            return

        label = labels_col.find_one({"productId": product_id, "reviewCycle": rc})
        assess = assess_col.find_one({"productId": product_id, "reviewCycle": rc})

        if not label or "y" not in label or assess is None:
            return

        x, cx, model_version = _extract_x_cx_from_assess(assess)
        if not x:
            return  # no usable breakdown

        ds_doc = {
            "productId": product_id,
            "reviewCycle": rc,
            "X": x + cx,
            "y": float(label["y"]),
            "ts": time.time(),
            "modelVersion": model_version,
        }

        # Insert-only (the live path already keeps it fresh if label was present then)
        dataset_col.update_one(
            {"productId": product_id, "reviewCycle": rc},
            {"$setOnInsert": ds_doc},
            upsert=True,
        )
        bt.logging.info(
            f"[Sync] Backfilled dataset row for product={product_id} rc={rc}"
        )

    except Exception as e:
        bt.logging.error(f"[Sync] Failed dataset sync for product {p.get('id')}: {e}")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_page(page: int, limit: int = PAGE_SIZE) -> Dict[str, Any]:
    url = f"{BASE}/products?page={page}&limit={limit}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_raw_and_label(p: Dict[str, Any]):
    # Raw
    raw_col.update_one({"id": p["id"]}, {"$set": p}, upsert=True)

    # Label (target)
    trust = p.get("trustScore")
    if trust is not None:
        labels_col.update_one(
            {
                "productId": p["id"],
                "reviewCycle": _rc(p),
            },
            {
                "$set": {
                    "y": float(trust),
                    "sourceUpdatedAt": p.get("updatedAt"),
                    "ts": time.time(),
                }
            },
            upsert=True,
        )
        # Attempt to sync a dataset row now that a label exists.
        _try_sync_dataset(p)


async def assess_product(p: Dict[str, Any]):
    """
    Runs the LLM assessment (your function). That function already persists the `breakdown`
    into DB. After completion, attempt to sync into cc_datasets if a label already exists.
    """

    # Try to read the saved breakdown (features) from cc_assessments:
    rc = _rc(p)
    doc = assess_col.find_one({"productId": p["id"], "reviewCycle": rc})
    if not doc:
        bt.logging.info(
            f"[Backfill] Generating assessment using LLM for product {p['id']}"
        )
        try:
            await generate_complete_assessment(SimpleProduct(p))
        except Exception as e:
            bt.logging.error(f"[assess] LLM failed for {p['id']}: {e}")

        doc = assess_col.find_one({"productId": p["id"], "reviewCycle": rc})
        if not doc:
            return

    # Try to sync a dataset row now that an assessment exists.
    _try_sync_dataset(p)


class SimpleProduct:
    """Lightweight adapter so you can pass to your LLM function if it expects attributes."""

    def __init__(self, d: Dict[str, Any]):
        self.__dict__.update(d)
        # For exact fields your function uses:
        self._id = d.get("_id")
        self.name = d.get("name")
        self.description = d.get("description")
        self.url = d.get("url")
        self.category = (
            d.get("category", {}).get("name")
            if isinstance(d.get("category"), dict)
            else d.get("category")
        )
        self.currentReviewCycle = d.get("currentReviewCycle")


async def main():
    ensure_indexes()
    # First discover total count
    first = fetch_page(1, limit=1)
    total = first["data"]["total"]
    pages = math.ceil(total / PAGE_SIZE)
    bt.logging.info(f"Found total={total}, pages={pages}")

    for page in range(1, pages + 1):
        data = fetch_page(page, PAGE_SIZE)
        products = data["data"]["products"]

        # Save raw + labels (+ opportunistic dataset sync)
        for p in products:
            save_raw_and_label(p)

        # Assess concurrently but gently (each task also tries dataset sync)
        tasks = [assess_product(p) for p in products]
        await asyncio.gather(*tasks)

        # Final sweep for this page (covers any races where both were present but sync missed)
        for p in products:
            _try_sync_dataset(p)

    upsert_meta("last_backfill_run", time.time())
    bt.logging.info("[DONE] Ingestion + assessment + dataset sync complete.")


if __name__ == "__main__":
    bt.logging.set_trace()
    asyncio.run(main())
