from __future__ import annotations
import os, math, asyncio, time, json
from typing import Dict, Any, List
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

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
                "reviewCycle": int(p.get("currentReviewCycle") or 1),
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


async def assess_product(p: Dict[str, Any]):
    """
    Runs the LLM assessment (your function). That function already persists the `breakdown`
    into DB. We’ll wait a moment and then read the breakdown back to mirror into cc_datasets
    **if** a label already exists. If label comes later, a later run will join it then.
    """
    try:
        await generate_complete_assessment(p)
    except Exception as e:
        print(f"[assess] LLM failed for {p['id']}: {e}")

    # Try to read the saved breakdown (features) from cc_assessments:
    doc = assess_col.find_one(
        {"productId": p["id"], "reviewCycle": int(p.get("currentReviewCycle") or 1)}
    )
    if not doc:
        return

    X = [float(doc.get("breakdown", {}).get(k, 0.0)) for k in METRICS]

    # If we already have label, materialize a training row
    lbl = labels_col.find_one(
        {"productId": p["id"], "reviewCycle": int(p.get("currentReviewCycle") or 1)}
    )
    if lbl and ("y" in lbl):
        dataset_col.update_one(
            {
                "productId": p["id"],
                "reviewCycle": int(p.get("currentReviewCycle") or 1),
            },
            {"$set": {"X": X, "y": float(lbl["y"]), "ts": time.time()}},
            upsert=True,
        )


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
    print(f"Found total={total}, pages={pages}")

    for page in range(1, pages + 1):
        data = fetch_page(page, PAGE_SIZE)
        products = data["data"]["products"]
        # Save raw + labels
        for p in products:
            save_raw_and_label(p)

        # Assess concurrently but gently
        tasks = [assess_product(p) for p in products]
        await asyncio.gather(*tasks)

    upsert_meta("last_backfill_run", time.time())
    print("[DONE] Ingestion + assessment complete.")


if __name__ == "__main__":
    asyncio.run(main())
