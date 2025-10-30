from __future__ import annotations
import re, json
from typing import Any, Dict, Optional
import numbers

from ..config import LOG, METRICS

_JSON_BLOCK_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _norm_key(k):
    if not isinstance(k, str):
        return k
    s = k.strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        s = s[1:-1].strip()
    return s


def _normalize_keys(obj):
    if isinstance(obj, dict):
        return {_norm_key(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(v) for v in obj]
    return obj


def _to_float_safe(x, default: float) -> float:
    if isinstance(x, numbers.Real):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return default
    return default


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
        data = json.loads(block)
    except Exception:
        try:
            from json_repair import repair_json

            data = json.loads(repair_json(block))
        except Exception:
            return {}
    return _normalize_keys(data)


def _mk_default_item() -> Dict[str, Any]:
    return {
        "score": 5.0,
        "rationale": "Evidence limited; leaning on general model knowledge.",
        "citations": ["model_knowledge"],
        "confidence": 0.4,
    }


def coerce_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    data = _normalize_keys(data or {})
    out = dict(data)
    breakdown = out.get("breakdown", {})

    # tolerate weird shapes
    if isinstance(breakdown, list):
        breakdown = {
            m: (breakdown[i] if i < len(breakdown) else {})
            for i, m in enumerate(METRICS)
        }
    elif not isinstance(breakdown, dict):
        breakdown = {}

    fixed = {}
    for m in METRICS:
        item = breakdown.get(m) or _mk_default_item()
        if not isinstance(item, dict):
            item = _mk_default_item()
        else:
            item = _normalize_keys(item)

        score = max(0.0, min(10.0, _to_float_safe(item.get("score", 5.0), 5.0)))
        conf = max(0.0, min(1.0, _to_float_safe(item.get("confidence", 0.5), 0.5)))

        rationale = (
            str(item.get("rationale", "")).strip()
            or "No strong evidence; provisional assessment."
        )[:800]

        cits = item.get("citations") or []
        if not isinstance(cits, list):
            cits = [str(cits)]
        cits = [str(c).strip() for c in cits if str(c).strip()]
        if not cits:
            cits = ["model_knowledge"]
        if len(cits) > 5:
            cits = cits[:5]

        fixed[m] = {
            "score": score,
            "rationale": rationale,
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
    total = sum(weights[k] * fixed[k]["score"] for k in weights)
    overall = max(0.0, min(100.0, total * 10.0))

    review = (
        str(out.get("review", "")).strip()
        or "Preliminary review based on available evidence."
    )[:140]

    kws = out.get("keywords", [])
    if not isinstance(kws, list):
        kws = [str(kws)]
    kws = [str(k).strip() for k in kws if str(k).strip()]
    if len(kws) < 3:
        kws = (kws + ["crypto", "defi", "risk"])[:3]
    elif len(kws) > 7:
        kws = kws[:7]

    return {
        "breakdown": fixed,
        "overall_score": overall,
        "review": review,
        "keywords": kws,
    }
