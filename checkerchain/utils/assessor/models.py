from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field, conlist


class BreakdownItem(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: str
    citations: conlist(str, min_length=1, max_length=5)
    confidence: float = Field(ge=0, le=1)


class AssessmentModel(BaseModel):
    breakdown: Dict[str, BreakdownItem]
    overall_score: float = Field(ge=0, le=100)
    review: str = Field(max_length=140)
    keywords: conlist(str, min_length=3, max_length=7)
