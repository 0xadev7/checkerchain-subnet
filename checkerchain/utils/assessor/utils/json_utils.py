from __future__ import annotations
import re, json
from typing import Any, Dict, Optional
from ..config import LOG, METRICS

_JSON_BLOCK_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    txt = text.strip()
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```\s*$", "", txt)
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass
    cands = _JSON_BLOCK_RE.findall(txt)
    for block in sorted(cands, key=len, reverse=True):
        try:
            json.loads(block)
            return block
        except Exception:
            continue
    return None


def parse_or_repair_json(txt: str) -> dict:
    block = extract_json_block(txt) or txt
    try:
        return json.loads(block)
    except Exception:
        try:
            from json_repair import repair_json

            fixed = repair_json(block)
            return json.loads(fixed)
        except Exception:
            return {}


def _mk_default_item() -> Dict[str, Any]:
    return {
        "score": 5.0,
        "rationale": "Evidence limited; leaning on general model knowledge.",
        "citations": ["model_knowledge"],
        "confidence": 0.4,
    }


def coerce_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data or {})
    breakdown = dict(out.get("breakdown", {}))

    for m in METRICS:
        item = breakdown.get(m) or _mk_default_item()
        score = max(0.0, min(10.0, float(item.get("score", 5.0))))
        rationale = (
            str(item.get("rationale", "")).strip()
            or "No strong evidence; provisional assessment."
        )
        cits = item.get("citations") or []
        if not isinstance(cits, list):
            cits = [str(cits)]
        cits = [str(c).strip() for c in cits if str(c).strip()]
        if not cits:
            cits = ["model_knowledge"]
        if len(cits) > 5:
            cits = cits[:5]
        conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        breakdown[m] = {
            "score": score,
            "rationale": rationale[:800],
            "citations": cits,
            "confidence": conf,
        }

    weights = {
        "project": 0.12,
        "utility": 0.10,
        "userbase": 0.10,
        "team": 0.12,
        "security": 0.12,
        "tokenomics": 0.12,
        "marketing": 0.10,
        "partnerships": 0.10,
        "roadmap": 0.07,
        "clarity": 0.05,
    }
    total = sum(weights[k] * float(breakdown[k]["score"]) for k in weights)
    overall = max(0.0, min(100.0, total * 10.0))

    review = (
        str(out.get("review", "")).strip()
        or "Preliminary review based on available evidence."
    )
    review = review[:140]

    kws = out.get("keywords", [])
    if not isinstance(kws, list):
        kws = [str(kws)]
    kws = [str(k).strip() for k in kws if str(k).strip()]
    if len(kws) < 3:
        kws = (kws + ["crypto", "defi", "risk"])[:3]
    elif len(kws) > 7:
        kws = kws[:7]

    return {
        "breakdown": breakdown,
        "overall_score": overall,
        "review": review,
        "keywords": kws,
    }
