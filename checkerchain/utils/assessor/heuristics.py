from __future__ import annotations
from typing import Any, List


def should_research(product: Any) -> bool:
    text = f"{getattr(product,'name','')} {getattr(product,'description','')} {getattr(product,'url','')} {getattr(product,'category','')}".lower()
    return len(text) > 40


def format_evidence(notes: List[dict]) -> str:
    if not notes:
        return "No external evidence collected."
    lines = []
    for n in notes:
        src = n.get("url", "")
        title = n.get("title", "")
        snippet = (n.get("snippet") or n.get("text") or "")[:400]
        lines.append(f"- {title} — {src}\n  {snippet}")
    return "\n".join(lines)
