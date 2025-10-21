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
    docs = dedupe_docs(docs)

    # 2) map over chunks (parallel)
    mapped = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks[:6]):  # soft cap
            sid = f"{d['id']}-{i}"
            mapped.append((sid, ch, d))

    # NB: Parallelize in your infra; here sequential for clarity
    arrays = []
    for sid, ch, d in mapped:
        try:
            arr = await map_extract(llm_small, ch, sid)
            arrays.append(arr)
        except Exception:
            pass

    # 3) reduce
    merged = await reduce_facts(llm_small, arrays, max_facts=8)

    # 4) build source table
    keep_source_ids = {
        f["evidence"]["source_id"].split("-")[0]
        for f in merged.get("facts", [])
        if f.get("evidence")
    }
    sources = []
    for d in docs:
        if d["id"] in keep_source_ids:
            sources.append(
                {"id": d["id"], "url": d.get("url", ""), "title": d.get("title", "")}
            )

    # 5) enforce token budget by trimming lower-priority axes
    def pack_tokens(facts: List[Dict[str, Any]]) -> int:
        return num_tokens(json.dumps({"facts": facts[:], "sources": sources}))

    facts = merged.get("facts", [])
    # axis priority: security, team, tokenomics, partnerships, utility, userbase, roadmap, clarity, marketing, project
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
    by_axis = {ax: [f for f in facts if f.get("axis") == ax] for ax in priority}
    trimmed = []
    for ax in priority:
        for f in by_axis.get(ax, [])[:8]:
            trimmed.append(f)
            if pack_tokens(trimmed) > budget_tokens:
                trimmed.pop()
                break

    return {"facts": trimmed, "sources": sources}
