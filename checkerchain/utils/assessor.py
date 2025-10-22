import json
import re
import asyncio
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, conlist, model_validator, ValidationError
import bittensor as bt

from checkerchain.database.mongo import METRICS

# -------------------- Models --------------------


class BreakdownItem(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: str
    citations: conlist(str, min_length=1, max_length=5)
    confidence: float = Field(ge=0, le=1)  # 0..1


class Assessment(BaseModel):
    breakdown: Dict[str, BreakdownItem]
    overall_score: float = Field(ge=0, le=100)
    review: str = Field(max_length=140)
    keywords: conlist(str, min_length=3, max_length=7)

    @model_validator(mode="after")
    def check_all_metrics(self) -> "Assessment":
        """Ensure all 10 required metrics are present."""
        missing = [m for m in METRICS if m not in self.breakdown]
        if missing:
            raise ValueError(f"Missing required metrics: {missing}")
        return self


# -------------------- Prompts --------------------

SYSTEM = """You are an AI crypto analyst.
You MUST return one JSON object with these top-level keys ONLY: breakdown, overall_score, review, keywords.

Rules:
- Use provided facts and sources when available. If evidence is insufficient for a metric, you MAY use general knowledge but MUST:
  - include "model_knowledge" in citations,
  - lower confidence (e.g., 0.3–0.6),
  - avoid inventing specific, unverifiable claims.
- Prefer recent, reputable sources. If sources conflict, note briefly in rationale and prefer more credible/recent.
- breakdown must contain ALL metrics: ["project","userbase","utility","security","team","tokenomics","marketing","roadmap","clarity","partnerships"].
- Each metric item schema:
  {"score": 0..10, "rationale": str, "citations": [1..5 strings], "confidence": 0..1}
- overall_score is a 0..100 weighted average you compute (server may override weights later).
- review must be <= 140 characters (a single tight sentence).
- keywords must have 3..7 concise strings.

Return JSON ONLY. No backticks, no prose.
"""


def build_user_prompt(product, fact_pack):
    return f"""
Product:
- Name: {product.name}
- Description: {product.description}
- Website: {product.url}
- Category: {product.category}

Facts (with source_ids):
{json.dumps(fact_pack.get("facts", []), ensure_ascii=False)}

Sources (id→url):
{json.dumps(fact_pack.get("sources", []), ensure_ascii=False)}

Required metrics: {", ".join(METRICS)}.
Compute overall_score as a weighted average you deem reasonable.
Keep review <= 140 chars and keywords 3–7 items.
Return one JSON object with keys: breakdown, overall_score, review, keywords.
"""


# -------------------- Helpers --------------------

_JSON_BLOCK_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> Optional[str]:
    """Find the largest JSON-looking block and parseable chunk."""
    if not text:
        return None
    txt = text.strip()
    # common fence cleanup
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```\s*$", "", txt)
    # direct parse
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass
    # find candidate blocks
    cands = _JSON_BLOCK_RE.findall(txt)
    for block in sorted(cands, key=len, reverse=True):
        try:
            json.loads(block)
            return block
        except Exception:
            continue
    return None


def _coerce_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all metrics exist and fields are sane; add model_knowledge if empty citations."""
    out = {k: v for k, v in data.items()}
    breakdown = dict(out.get("breakdown", {}))

    def mk_default_item() -> Dict[str, Any]:
        return {
            "score": 5.0,
            "rationale": "Evidence limited; leaning on general model knowledge.",
            "citations": ["model_knowledge"],
            "confidence": 0.4,
        }

    for m in METRICS:
        item = breakdown.get(m) or mk_default_item()
        # normalize fields
        score = float(item.get("score", 5.0))
        score = max(0.0, min(10.0, score))
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
        conf = float(item.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))

        breakdown[m] = {
            "score": score,
            "rationale": rationale[:800],  # keep it tight
            "citations": cits,
            "confidence": conf,
        }

    # overall_score guard
    overall = float(
        out.get(
            "overall_score",
            sum(breakdown[m]["score"] for m in METRICS) / len(METRICS) * 10,
        )
    )
    overall = max(0.0, min(100.0, overall))

    # review & keywords guard
    review = str(out.get("review", "")).strip()
    if not review:
        review = "Preliminary review based on available evidence."
    review = review[:140]

    kws = out.get("keywords", [])
    if not isinstance(kws, list):
        kws = [str(kws)]
    kws = [str(k).strip() for k in kws if str(k).strip()]
    if len(kws) < 3:
        # synthesize minimal keywords from metrics
        kws = (kws + ["crypto", "defi", "risk"])[:3]
    elif len(kws) > 7:
        kws = kws[:7]

    return {
        "breakdown": breakdown,
        "overall_score": overall,
        "review": review,
        "keywords": kws,
    }


async def _ainvoke_json(
    llm, system: str, user: str, retries: int = 2
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            res = await llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            block = _extract_json(res.content)
            if not block:
                raise ValueError("No parseable JSON returned by model.")
            obj = json.loads(block)
            if not isinstance(obj, dict):
                raise ValueError("Top-level JSON must be an object.")
            # ensure the four keys exist at least
            for k in ("breakdown", "overall_score", "review", "keywords"):
                if k not in obj:
                    # allow later coercion, but keep explicit structure
                    obj.setdefault(
                        k,
                        (
                            {}
                            if k == "breakdown"
                            else (
                                []
                                if k == "keywords"
                                else 0 if k == "overall_score" else ""
                            )
                        ),
                    )
            return obj
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.25 * (attempt + 1))
    raise last_err or RuntimeError("Model failed to produce valid JSON.")


# -------------------- Main entry --------------------


async def run_assessor(llm_big, product, fact_pack) -> Dict[str, Any]:
    raw = await _ainvoke_json(
        llm_big,
        SYSTEM,
        build_user_prompt(product, fact_pack),
        retries=2,
    )

    # Coerce & fill any gaps before strict validation
    coerced = _coerce_with_defaults(raw)

    # Final strict validation; if it fails, try to salvage by injecting defaults
    try:
        parsed = Assessment.model_validate(coerced).model_dump()
    except ValidationError as e:
        # last-resort: ensure all metrics present + force ranges again
        salvaged = _coerce_with_defaults(coerced)
        parsed = Assessment.model_validate(salvaged).model_dump()

    return parsed
