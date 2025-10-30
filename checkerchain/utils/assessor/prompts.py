from __future__ import annotations
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

SYSTEM_JINJA = """You are an AI crypto analyst with tool use.
When uncertain, use tools to search & fetch sources. Cite with URLs.
If no strong source, include "model_knowledge" in citations and set confidence ≤ 0.6.

OUTPUT CONTRACT (MANDATORY):
- Return ONE JSON object ONLY with keys: breakdown, overall_score, review, keywords.
- No markdown, no backticks, no commentary.
- Each breakdown item: { "score": 0..10, "rationale": short (≤ 4 sentences), "citations": 1..5 strings (URLs or "model_knowledge"), "confidence": 0..1 }.
- Review ≤ 140 chars. 3–7 keywords.

Never reveal internal reasoning. Summarize briefly in 'rationale' only.
"""

USER_JINJA = """Product:
Name: {{ name }}
Description: {{ description }}
Website: {{ url }}
Category: {{ category }}

Evidence (from web, if any):
{{ evidence }}

Required metrics: {{ metrics }}
Rules:
- For each metric: score (0..10), rationale, citations (1..5 strings), confidence (0..1).
- Prefer recent/reputable sources; if conflicting, note briefly in rationale.
- If a metric lacks strong evidence, add "model_knowledge" in citations and set confidence ≤ 0.6.
- Keep review ≤ 140 chars; keywords 3–7 concise strings.
Return JSON ONLY.
"""

system_t = SystemMessagePromptTemplate.from_template(
    SYSTEM_JINJA, template_format="jinja2"
)
user_t = HumanMessagePromptTemplate.from_template(USER_JINJA, template_format="jinja2")

SCORING_PROMPT = ChatPromptTemplate.from_messages([system_t, user_t])
