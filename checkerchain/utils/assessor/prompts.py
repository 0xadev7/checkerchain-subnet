from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are an AI crypto analyst with tool use.
When uncertain, use tools to search & fetch sources. Cite with URLs.
If no strong source, include "model_knowledge" in citations and set confidence ≤ 0.6.

OUTPUT CONTRACT (MANDATORY):
- Return ONE JSON object ONLY with keys: breakdown, overall_score, review, keywords.
- No markdown, no backticks, no commentary.
- Each breakdown item: { "score": 0..10, "rationale": short (≤ 4 sentences), "citations": 1..5 strings (URLs or "model_knowledge"), "confidence": 0..1 }.
- Review ≤ 140 chars. 3–7 keywords.

Never reveal internal reasoning. Summarize briefly in 'rationale' only.
"""

SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "user",
            """Product:
Name: {name}
Description: {description}
Website: {url}
Category: {category}

Evidence (from web, if any):
{evidence}

Required metrics: {metrics}
Rules:
- For each metric: score (0..10), rationale, citations (1..5 strings), confidence (0..1).
- Prefer recent/reputable sources; if conflicting, note briefly in rationale.
- If a metric lacks strong evidence, add "model_knowledge" in citations and set confidence ≤ 0.6.
- Keep review ≤ 140 chars; keywords 3–7 concise strings.
Return JSON ONLY.
""",
        ),
    ]
)
