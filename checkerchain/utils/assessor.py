from __future__ import annotations
import os, re, json, asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

from pydantic import BaseModel, Field, conlist, ValidationError
from langchain_core.tools import tool, Tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ---------- Config ----------
USER_AGENT = "CheckerChainAssessor/1.0 (+https://checkerchain.com/)"
ABSOLUTE_MAX_BYTES = 450_000
FETCH_MAX_BYTES = 220_000
SEARCH_TOP_K = 6
FETCH_TOP_N = 3

METRICS = [
    "project",
    "utility",
    "userbase",
    "team",
    "security",
    "tokenomics",
    "marketing",
    "partnerships",
    "roadmap",
    "clarity",
]

# Optional keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()


# ---------- Output schema ----------
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


# ---------- JSON parsing / extraction ----------
_JSON_BLOCK_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json_block(text: str) -> Optional[str]:
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


def _parse_or_repair_json(txt: str) -> dict:
    block = _extract_json_block(txt) or txt
    try:
        return json.loads(block)
    except Exception:
        try:
            from json_repair import repair_json

            fixed = repair_json(block)
            return json.loads(fixed)
        except Exception:
            return {}


# ---------- Coercion & defaults ----------
def _mk_default_item() -> Dict[str, Any]:
    return {
        "score": 5.0,
        "rationale": "Evidence limited; leaning on general model knowledge.",
        "citations": ["model_knowledge"],
        "confidence": 0.4,
    }


def _coerce_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
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


# ---------- Web tools: Tavily → Brave → DuckDuckGo ----------
async def _tavily_search(query: str, k: int = 5) -> List[dict]:
    if not TAVILY_API_KEY:
        return []
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": k}
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
        async with sess.post(url, json=payload, timeout=25) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    results = []
    for r in (data.get("results") or [])[:k]:
        results.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or "",
            }
        )
    return [r for r in results if r["url"]]


async def _brave_search(query: str, k: int = 5) -> List[dict]:
    if not BRAVE_API_KEY:
        return []
    endpoint = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": k}
    headers = {"X-Subscription-Token": BRAVE_API_KEY, "User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as sess:
        async with sess.get(endpoint, params=params, timeout=25) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    web = (data.get("web") or {}).get("results") or []
    out = []
    for r in web[:k]:
        out.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("description") or "",
            }
        )
    return [r for r in out if r["url"]]


async def _ddg_search(query: str, k: int = 5) -> List[dict]:
    url = f"https://duckduckgo.com/html/?q={query}"
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
        async with sess.get(url, timeout=25) as resp:
            html = await resp.text()
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for r in soup.select(".result__body")[:k]:
        link = r.select_one(".result__a")
        snip = r.select_one(".result__snippet")
        if link and link.get("href"):
            out.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": link.get("href"),
                    "snippet": snip.get_text(" ", strip=True) if snip else "",
                }
            )
    return out


async def _tiered_search_impl(query: str, k: int = SEARCH_TOP_K) -> List[dict]:
    q = query.strip()
    if not q:
        return []
    res = await _tavily_search(q, k)
    if not res:
        res = await _brave_search(q, k)
    if not res:
        res = await _ddg_search(q, k)
    return res


@tool
async def web_search(query: str, k: int = SEARCH_TOP_K) -> List[dict]:
    """
    Tiered web search. Returns a list of {title, url, snippet}.
    Order of providers: Tavily → Brave → DuckDuckGo.
    """
    return await _tiered_search_impl(query, k)


@tool
async def web_fetch(url: str, max_bytes: int = FETCH_MAX_BYTES) -> dict:
    """
    Fetch a URL and return {url, title, text}. Truncates to max_bytes.
    """
    cap = min(max_bytes, ABSOLUTE_MAX_BYTES)
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
        async with sess.get(url, timeout=30) as resp:
            content = await resp.read()
    content = content[:cap]
    html = content.decode(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    title = soup.title.get_text(strip=True) if soup.title else url
    return {"url": url, "title": title, "text": text[:cap]}


# ---------- Prompts ----------
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


# ---------- Heuristics ----------
def _should_research(product: Any) -> bool:
    text = f"{getattr(product,'name','')} {getattr(product,'description','')} {getattr(product,'url','')} {getattr(product,'category','')}".lower()
    return len(text) > 40


def _format_evidence(notes: List[dict]) -> str:
    if not notes:
        return "No external evidence collected."
    lines = []
    for n in notes:
        src = n.get("url", "")
        title = n.get("title", "")
        snippet = (n.get("snippet") or n.get("text") or "")[:400]
        lines.append(f"- {title} — {src}\n  {snippet}")
    return "\n".join(lines)


# ---------- Graph ----------
class AssessorState(Dict[str, Any]): ...


def _build_graph(llm, tools: List[Tool]):
    g = StateGraph(AssessorState)

    async def decide(state: AssessorState):
        product = state["product"]
        state["do_research"] = _should_research(product)
        return state

    async def research(state: AssessorState):
        if not state.get("do_research"):
            state["evidence"] = []
            return state
        p = state["product"]
        seed_url = getattr(p, "url", "") or ""
        q = f'{getattr(p, "name", "")} {getattr(p, "category", "")} crypto token whitepaper roadmap security team site:{seed_url}'
        search_res = await web_search.ainvoke({"query": q, "k": SEARCH_TOP_K})
        top = search_res[:FETCH_TOP_N] if isinstance(search_res, list) else []
        fetched = []
        for r in top:
            try:
                doc = await web_fetch.ainvoke({"url": r["url"]})
                fetched.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                        "text": doc.get("text", ""),
                    }
                )
            except Exception:
                continue
        state["evidence"] = fetched if fetched else top
        return state

    async def score(state: AssessorState):
        p = state["product"]
        evidence = _format_evidence(state.get("evidence", []))
        prompt = SCORING_PROMPT.format_messages(
            name=getattr(p, "name", ""),
            description=getattr(p, "description", ""),
            url=getattr(p, "url", ""),
            category=getattr(p, "category", ""),
            evidence=evidence,
            metrics=", ".join(METRICS),
        )
        msg = await llm.ainvoke(prompt)
        state["raw_text"] = msg.content if isinstance(msg, AIMessage) else str(msg)
        return state

    async def validate(state: AssessorState):
        txt = state.get("raw_text", "")
        obj = _parse_or_repair_json(txt)
        coerced = _coerce_with_defaults(obj)
        try:
            parsed = AssessmentModel.model_validate(coerced).model_dump()
        except ValidationError:
            parsed = AssessmentModel.model_validate(
                _coerce_with_defaults(coerced)
            ).model_dump()
        state["final"] = parsed
        return state

    g.add_node("decide", decide)
    g.add_node("research", research)
    g.add_node("score", score)
    g.add_node("validate", validate)

    g.set_entry_point("decide")
    g.add_edge("decide", "research")
    g.add_edge("research", "score")
    g.add_edge("score", "validate")
    g.add_edge("validate", END)
    return g.compile()


# ---------- Public entry point ----------
async def run_assessor(llm_big, product) -> Dict[str, Any]:
    """
    Drop-in entry point:
        parsed = await run_assessor(llm_big=llm_big, product=product)
    """
    llm = llm_big.bind_tools([web_search, web_fetch])
    graph = _build_graph(llm, [web_search, web_fetch])
    state = {"product": product}
    result = await graph.ainvoke(state)
    return result["final"]
