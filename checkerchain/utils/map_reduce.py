import json
import math
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

# ---------- Prompts ----------
MAP_PROMPT = """You extract FACTS needed to assess a DeFi/crypto product.
Return a JSON array ONLY (no prose). Each item:
{
  "claim": <atomic, verifiable fact from the chunk>,
  "axis": one of ["project","userbase","utility","security","team","tokenomics","marketing","roadmap","clarity","partnerships","other"],
  "evidence": {"source_id": "<source_id passed in>", "snippet": "<verbatim phrase from the chunk>"}
}
Rules:
- Use only information present in the chunk. No opinions, no external info.
- Prefer exact quotes or very short paraphrases in "snippet".
- If unsure about axis, use "other".
- Output JSON ONLY.
"""

REDUCE_PROMPT = """You merge JSON arrays of facts.
Tasks:
- Remove duplicates/near-duplicates.
- When conflicting, choose items with clearer, more specific snippets.
- Keep up to {max_facts} facts PER AXIS (not total).
- Keep the same schema per item.
Output EXACTLY one JSON object: {"facts":[...]} (no prose).
"""

AXES = {
    "project",
    "userbase",
    "utility",
    "security",
    "team",
    "tokenomics",
    "marketing",
    "roadmap",
    "clarity",
    "partnerships",
    "other",
}


# ---------- Utils ----------
def extract_json_from_text(text: str) -> Optional[str]:
    """Grab the largest JSON block (helps when model adds stray prose)."""
    if text is None:
        return None
    # Try direct parse first
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # Fallback: find first {...} or [...] that parses
    candidates = []
    for m in re.finditer(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL):
        candidates.append(m.group(0))
    # Try longer candidates first
    for block in sorted(candidates, key=len, reverse=True):
        try:
            json.loads(block)
            return block
        except Exception:
            continue
    return None


def safe_json_loads(text: str, default):
    if not text:
        return default
    block = extract_json_from_text(text)
    if not block:
        return default
    try:
        return json.loads(block)
    except Exception:
        return default


def _sanitize_fact(
    raw: Dict[str, Any], default_source_id: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    claim = (raw.get("claim") or "").strip()
    if not claim:
        return None
    axis = str(raw.get("axis") or "other").strip().lower()
    if axis not in AXES:
        axis = "other"

    ev = raw.get("evidence") or {}
    if not isinstance(ev, dict):
        ev = {}
    source_id = str(ev.get("source_id") or default_source_id or "").strip()
    snippet = str(ev.get("snippet") or "").strip()
    # keep snippets concise to reduce tokens
    if len(snippet) > 450:
        snippet = snippet[:447] + "..."

    return {
        "claim": claim,
        "axis": axis,
        "evidence": {"source_id": source_id, "snippet": snippet},
    }


def _hash_fact(f: Dict[str, Any]) -> Tuple[str, str]:
    # normalize for dedupe: lowercased claim, axis
    return (f["claim"].strip().lower(), f["axis"])


def _dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple deterministic de-duplication by normalized (claim, axis).
    Chooses the item with the longer snippet (as proxy for specificity)."""
    best = {}
    for f in facts:
        key = _hash_fact(f)
        cur = best.get(key)
        if cur is None or len((f["evidence"]["snippet"] or "")) > len(
            (cur["evidence"]["snippet"] or "")
        ):
            best[key] = f
    return list(best.values())


async def _ainvoke_with_retry(llm, messages, attempts=2) -> Any:
    last = None
    for i in range(attempts):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            last = e
            await asyncio.sleep(0.25 * (i + 1))
    raise last if last else RuntimeError("LLM invocation failed")


# ---------- Map ----------
async def map_extract(llm, chunk_text: str, source_id: str) -> List[Dict[str, Any]]:
    """Always returns a *list* of sanitized fact dicts."""
    msg = f"{MAP_PROMPT}\n\n<chunk source_id='{source_id}'>\n{chunk_text}\n</chunk>"
    res = await _ainvoke_with_retry(
        llm,
        [
            SystemMessage(content="You are a precise fact extractor."),
            HumanMessage(content=msg),
        ],
    )
    data = safe_json_loads(res.content, default=[])
    # Accept either a list or an object with "facts"
    if isinstance(data, dict) and isinstance(data.get("facts"), list):
        data = data["facts"]
    if not isinstance(data, list):
        data = []

    sanitized: List[Dict[str, Any]] = []
    for item in data:
        fact = _sanitize_fact(item, default_source_id=source_id)
        if fact:
            sanitized.append(fact)

    return sanitized


# ---------- Reduce ----------
async def reduce_facts(
    llm, arrays: List[List[Dict[str, Any]]], max_facts: int = 8
) -> Dict[str, Any]:
    """Returns {"facts": [...]} with sanitized, deduped facts.
    Handles large inputs via chunking, plus local dedupe as a guardrail."""
    # Flatten & sanitize first (so we don’t send garbage upstream)
    flat: List[Dict[str, Any]] = []
    for i, arr in enumerate(arrays or []):
        if not isinstance(arr, list):
            continue
        for item in arr:
            fact = _sanitize_fact(
                item, default_source_id=item.get("evidence", {}).get("source_id", "")
            )
            if fact:
                flat.append(fact)

    if not flat:
        return {"facts": []}

    # Local dedupe before sending to LLM
    flat = _dedupe_facts(flat)

    # Chunk into reasonable batches to avoid context blowups (8–12k chars per batch)
    payload_items_per_batch = max(64, min(300, max_facts * 20))
    batches = [
        flat[i : i + payload_items_per_batch]
        for i in range(0, len(flat), payload_items_per_batch)
    ]

    merged: List[Dict[str, Any]] = []
    for bi, batch in enumerate(batches):
        combined = json.dumps(batch)
        res = await _ainvoke_with_retry(
            llm,
            [
                SystemMessage(content="You are a meticulous merger and deduper."),
                HumanMessage(
                    content=REDUCE_PROMPT.format(max_facts=max_facts)
                    + "\n\n"
                    + combined
                ),
            ],
        )
        obj = safe_json_loads(res.content, default={"facts": []})
        facts = obj.get("facts") if isinstance(obj, dict) else []
        if not isinstance(facts, list):
            facts = []
        # sanitize again & local dedupe
        batch_sanitized = []
        for f in facts:
            s = _sanitize_fact(
                f, default_source_id=f.get("evidence", {}).get("source_id", "")
            )
            if s:
                batch_sanitized.append(s)
        merged.extend(batch_sanitized)
        # Local dedupe after each batch
        merged = _dedupe_facts(merged)

    # Enforce per-axis cap locally (don’t trust the model to obey)
    per_axis: Dict[str, List[Dict[str, Any]]] = {ax: [] for ax in AXES}
    for f in merged:
        ax = f["axis"] if f["axis"] in AXES else "other"
        if len(per_axis[ax]) < max_facts:
            per_axis[ax].append(f)

    final = [f for ax in sorted(per_axis.keys()) for f in per_axis[ax]]
    return {"facts": final}
