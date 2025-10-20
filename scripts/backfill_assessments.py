# scripts/backfill_assessments.py
from __future__ import annotations
import os, asyncio, time, sys, math, argparse, json, re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

# Your project imports
# Make sure these resolve based on your repo layout
from checkerchain.miner.llm import generate_complete_assessment
from checkerchain.rlhf.db_mongo import RLHFMongo
from checkerchain.rlhf.constants import METRICS

API_BASE = os.getenv("CHECKERCHAIN_API", "https://api.checkerchain.com/api/v1/products")

# -------- Helpers --------


def mk_product_ns(p: dict) -> SimpleNamespace:
    """Create a lightweight object with attributes the LLM function expects."""
    # Fallbacks to keep it robust
    pid = str(p.get("_id") or p.get("id"))
    cat = p.get("category", {})
    category_name = cat.get("name") if isinstance(cat, dict) else str(cat)
    return SimpleNamespace(
        id=pid,
        name=p.get("name", ""),
        description=p.get("description", "") or "",
        url=p.get("url", "") or "",
        category=category_name or "",
        currentReviewCycle=int(p.get("currentReviewCycle", 1)),
    )


def extract_breakdown_from_response(resp: dict) -> Optional[dict]:
    """Expect 'breakdown' in the new return format; keep a fallback."""
    if not isinstance(resp, dict):
        return None
    # Preferred (after you patched llm.py)
    if "breakdown" in resp and isinstance(resp["breakdown"], dict):
        return resp["breakdown"]
    # If not patched yet, try to parse from review/keywords (unlikely). Give up cleanly.
    return None


# -------- API paging --------


async def fetch_page(
    session: aiohttp.ClientSession, page: int, limit: int
) -> List[dict]:
    params = {"page": page, "limit": limit}
    async with session.get(API_BASE, params=params, timeout=60) as r:
        r.raise_for_status()
        data = await r.json()
        return data.get("data", {}).get("products", []) or []


async def iter_reviewed_products(limit_per_page: int, max_pages: int) -> List[dict]:
    out: List[dict] = []
    async with aiohttp.ClientSession() as session:
        for page in range(1, max_pages + 1):
            items = await fetch_page(session, page, limit_per_page)
            if not items:
                break
            # keep only reviewed
            reviewed = [p for p in items if p.get("status") == "reviewed"]
            out.extend(reviewed)
            if len(items) < limit_per_page:
                break
    return out


# -------- Backfill worker --------


async def assess_one(
    db: RLHFMongo,
    prod: dict,
    sem: asyncio.Semaphore,
    dry_run: bool = False,
    save_even_if_exists: bool = False,
) -> dict:
    """Call generate_complete_assessment and store breakdown (if available)."""
    async with sem:
        pns = mk_product_ns(prod)
        key = {"product_id": pns.id, "review_cycle": int(pns.currentReviewCycle)}

        # Skip if already have breakdown (unless forcing)
        if not save_even_if_exists:
            existing = db.col_breakdowns.find_one(key)
            if existing:
                return {"id": pns.id, "skipped": "exists"}

        try:
            resp = await generate_complete_assessment(pns)
        except Exception as e:
            return {"id": pns.id, "error": f"llm_failed: {e}"}

        bd = extract_breakdown_from_response(resp)
        if not bd:
            return {"id": pns.id, "error": "no_breakdown_in_response"}

        # Normalize x vector
        x = [float(bd.get(k, 0.0)) for k in METRICS]
        x = list(np.clip(np.array(x, dtype=float), 0.0, 10.0))

        if dry_run:
            return {"id": pns.id, "ok": True, "dry_run": True, "x": x}

        try:
            db.save_breakdown(
                product_id=pns.id,
                review_cycle=int(pns.currentReviewCycle),
                x=x,
                model_version="v1",
            )
            return {"id": pns.id, "ok": True}
        except Exception as e:
            return {"id": pns.id, "error": f"save_failed: {e}"}


# -------- Main --------


async def main():
    ap = argparse.ArgumentParser(
        description="Backfill LLM assessments (breakdowns) for reviewed products."
    )
    ap.add_argument(
        "--limit", type=int, default=30, help="Items per page when fetching API"
    )
    ap.add_argument("--pages", type=int, default=50, help="Max pages to fetch")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent LLM workers")
    ap.add_argument("--dry-run", action="store_true", help="Do not persist to DB")
    ap.add_argument(
        "--force", action="store_true", help="Recompute/save even if breakdown exists"
    )
    args = ap.parse_args()

    db = RLHFMongo()
    products = await iter_reviewed_products(args.limit, args.pages)
    if not products:
        print(json.dumps({"status": "no-products"}))
        return

    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        assess_one(db, p, sem, dry_run=args.dry_run, save_even_if_exists=args.force)
        for p in products
    ]
    results: List[dict] = []
    # small batching to avoid giant gather memory spikes
    BATCH = 32
    for i in range(0, len(tasks), BATCH):
        chunk = tasks[i : i + BATCH]
        results.extend(await asyncio.gather(*chunk))

    # Summarize
    ok = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = [r for r in results if r.get("error")]
    summary = {
        "total_reviewed": len(products),
        "ok": ok,
        "skipped": skipped,
        "errors": len(errors),
        "error_examples": errors[:5],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
