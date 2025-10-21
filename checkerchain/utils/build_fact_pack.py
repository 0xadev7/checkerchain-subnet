import json
from typing import Dict, Any, List

from checkerchain.utils.context_compressor import num_tokens, chunk_text, dedupe_docs
from checkerchain.utils.map_reduce import map_extract, reduce_facts


async def build_fact_pack(
    llm_small, docs: List[Dict[str, Any]], budget_tokens: int = 2000
) -> Dict[str, Any]:
    """
    docs: [{"url":..., "text":..., "id":...}, ...]
    Returns: {"facts":[{"claim":...,"axis":...,"evidence":{"source_id":...,"snippet":...}}], "sources":[{"id":...,"url":...,"title":...}]}
    """
    # 1) clean & dedupe
    docs = dedupe_docs(docs or [])

    # 2) map over chunks (parallel)
    mapped = []
    for d in docs:
        if not d or "text" not in d or not d["text"]:
            continue
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks[:6]):  # soft cap
            sid = f"{d.get('id','unknown')}-{i}"
            mapped.append((sid, ch, d))

    # NB: Parallelize in your infra; here sequential for clarity
    arrays: List[Any] = []
    for sid, ch, d in mapped:
        try:
            arr = await map_extract(llm_small, ch, sid)
            if arr is not None:
                arrays.append(arr)
        except Exception:
            # swallow per-chunk failures
            pass

    # 3) reduce
    try:
        merged = await reduce_facts(llm_small, arrays, max_facts=8)
    except Exception:
        merged = None

    # --- Normalization helpers ------------------------------------------------

    def _coerce_facts_from_any(x: Any) -> List[Dict[str, Any]]:
        """
        Accepts: None | list | dict | weird
        Returns: list[dict] of facts
        """
        if x is None:
            return []
        # If reducer gave a dict, try common keys
        if isinstance(x, dict):
            if isinstance(x.get("facts"), list):
                return x["facts"]
            for alt in ("items", "claims", "result", "data"):
                v = x.get(alt)
                if isinstance(v, list):
                    return v
            # nothing recognizable
            return []
        # If reducer produced a plain list, assume it's facts-like
        if isinstance(x, list):
            return x
        return []

    def _flatten_facts_from_maps(arrs: List[Any]) -> List[Dict[str, Any]]:
        """
        Try to salvage facts directly from map outputs when reduce is unusable.
        We expect each `arr` to be either:
          - {"facts":[...]} or list of facts, or
          - {"items":[...]} etc.
        """
        out: List[Dict[str, Any]] = []
        for a in arrs:
            out.extend(_coerce_facts_from_any(a))
        return out

    def _sanitize_fact(f: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure minimal schema: claim:str, axis:str, evidence:{source_id, snippet}
        """
        claim = f.get("claim") or f.get("text") or f.get("statement") or ""
        axis = f.get("axis") or "other"

        ev = f.get("evidence") or {}
        if not isinstance(ev, dict):
            ev = {}
        source_id = ev.get("source_id") or f.get("source_id") or ""
        snippet = ev.get("snippet") or f.get("snippet") or ""

        return {
            "claim": str(claim).strip(),
            "axis": str(axis).strip() if axis else "other",
            "evidence": {
                "source_id": str(source_id) if source_id is not None else "",
                "snippet": str(snippet).strip(),
            },
        }

    # --- Normalize reducer output; fallback to mapped results -----------------

    facts = _coerce_facts_from_any(merged)
    if not facts:
        # Fallback: flatten from map phase
        facts = _flatten_facts_from_maps(arrays)

    # Sanitize shape
    facts = [_sanitize_fact(f) for f in facts if isinstance(f, dict)]

    # 4) build source table from what survived
    keep_source_ids = {
        (f.get("evidence") or {}).get("source_id", "").split("-")[0]
        for f in facts
        if f.get("evidence")
    }
    sources = []
    for d in docs:
        if d.get("id") in keep_source_ids:
            sources.append(
                {"id": d["id"], "url": d.get("url", ""), "title": d.get("title", "")}
            )

    # 5) enforce token budget by trimming lower-priority axes
    def pack_tokens(facts_subset: List[Dict[str, Any]]) -> int:
        try:
            return num_tokens(json.dumps({"facts": facts_subset, "sources": sources}))
        except Exception:
            # ultra-defensive fallback; rough estimate
            return sum(len(json.dumps(f)) for f in facts_subset) // 4

    # axis priority: security, team, tokenomics, partnerships, utility, userbase, roadmap, clarity, marketing, project, other
    priority = [
        "security",
        "team",
        "tokenomics",
        "partnerships",
        "utility",
        "userbase",
        "roadmap",
        "clarity",
        "marketing",
        "project",
        "other",
    ]

    # bucket by axis (unknowns default to "other")
    by_axis: Dict[str, List[Dict[str, Any]]] = {ax: [] for ax in priority}
    for f in facts:
        ax = f.get("axis") or "other"
        if ax not in by_axis:
            ax = "other"
        by_axis[ax].append(f)

    trimmed: List[Dict[str, Any]] = []
    # take up to 8 per axis, in priority order, but stop when token budget would be exceeded
    for ax in priority:
        for f in by_axis.get(ax, [])[:8]:
            candidate = trimmed + [f]
            if pack_tokens(candidate) <= budget_tokens:
                trimmed.append(f)
            else:
                # once we exceed, skip remaining in this axis and continue with next axes
                break

    return {"facts": trimmed, "sources": sources}
