import json
from typing import Dict, Any
import re

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, conlist, validator

METRICS = [
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
]


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


SYSTEM = """You are an AI crypto analyst.
Use provided facts and sources when available. If evidence is insufficient for a metric, you MAY use your general knowledge but MUST:
- add "model_knowledge" to citations,
- reduce confidence,
- avoid inventing specific, unverifiable claims.
Prefer recent, reputable sources. If sources conflict, state that briefly in rationale and use the most credible/recent.
Return ONLY the JSON per the schema.
"""


def build_user_prompt(product, fact_pack):
    return f"""
Product:
- Name: {product.name}
- Description: {product.description}
- Website: {product.url}
- Category: {product.category}

Facts (with source_ids):
{json.dumps(fact_pack["facts"][:], ensure_ascii=False)}

Sources (id→url):
{json.dumps(fact_pack["sources"][:], ensure_ascii=False)}

Required metrics: {", ".join(METRICS)}.
Compute overall_score as a weighted average (server may override weights later). Keep review <=140 chars and keywords 3-7 items.
"""


async def run_assessor(llm_big, product, fact_pack) -> Dict[str, Any]:
    result = await llm_big.ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=build_user_prompt(product, fact_pack)),
        ]
    )
    txt = result.content.strip()
    txt = re.sub(r"^```json\s*|\s*```$", "", txt)
    data = json.loads(txt)
    # Validate & coerce
    parsed = Assessment.model_validate(data).model_dump()
    return parsed
