from __future__ import annotations
import os, re, json, asyncio, time, uuid, logging
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

from pydantic import BaseModel, Field, conlist, ValidationError
from langchain_core.tools import tool, Tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# Prefer bittensor logger if available
try:
    import bittensor as bt

    _LOG = bt.logging
except Exception:
    _LOG = logging.getLogger("assessor")
    if not _LOG.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _LOG.addHandler(_h)
    _LOG.setLevel(logging.INFO)

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


# ---------- Verbose callback handler ----------
class AssessorCallbackHandler(BaseCallbackHandler):
    def __init__(self, run_id: str, verbose: bool = False):
        self.run_id = run_id
        self.verbose = verbose

    # LLM events
    def on_llm_start(self, serialized, prompts, **kwargs):
        if not self.verbose:
            return
        model = (serialized or {}).get("id", "llm")
        _LOG.info(
            f"[assessor:{self.run_id}] LLM start: {model} | prompts={len(prompts)}"
        )

    def on_llm_end(self, response, **kwargs):
        if not self.verbose:
            return
        usage = getattr(response, "llm_output", {}) or {}
        _LOG.info(f"[assessor:{self.run_id}] LLM end: usage={usage}")

    def on_llm_error(self, error, **kwargs):
        _LOG.error(f"[assessor:{self.run_id}] LLM error: {error}")

    # Tool events
    def on_tool_start(self, serialized, input_str, **kwargs):
        if not self.verbose:
            return
        name = (serialized or {}).get("name", "tool")
        _LOG.info(f"[assessor:{self.run_id}] Tool start: {name} | input={input_str}")

    def on_tool_end(self, output, **kwargs):
        if not self.verbose:
            return
        # Truncate noisy outputs
        summary = str(output)
        if len(summary) > 500:
            summary = summary[:500] + "...[trunc]"
        _LOG.info(f"[assessor:{self.run_id}] Tool end: {summary}")

    def on_tool_error(self, error, **kwargs):
        _LOG.error(f"[assessor:{self.run_id}] Tool error: {error}")


# ---------- Web tools: Tavily → Brave → DuckDuckGo ----------
async def _tavily_search(query: str, k: int = 5, run_id: str = "-") -> List[dict]:
    if not TAVILY_API_KEY:
        return []
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": k}
    t0 = time.time()
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
            async with sess.post(url, json=payload, timeout=25) as resp:
                if resp.status != 200:
                    _LOG.warning(f"[assessor:{run_id}] Tavily HTTP {resp.status}")
                    return []
                data = await resp.json()
        dt = (time.time() - t0) * 1000
        _LOG.info(f"[assessor:{run_id}] Tavily search ok in {dt:.1f} ms")
    except Exception as e:
        _LOG.error(f"[assessor:{run_id}] Tavily error: {e}")
        return []
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


async def _brave_search(query: str, k: int = 5, run_id: str = "-") -> List[dict]:
    if not BRAVE_API_KEY:
        return []
    endpoint = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": k}
    headers = {"X-Subscription-Token": BRAVE_API_KEY, "User-Agent": USER_AGENT}
    t0 = time.time()
    try:
        async with aiohttp.ClientSession(headers=headers) as sess:
            async with sess.get(endpoint, params=params, timeout=25) as resp:
                if resp.status != 200:
                    _LOG.warning(f"[assessor:{run_id}] Brave HTTP {resp.status}")
                    return []
                data = await resp.json()
        dt = (time.time() - t0) * 1000
        _LOG.info(f"[assessor:{run_id}] Brave search ok in {dt:.1f} ms")
    except Exception as e:
        _LOG.error(f"[assessor:{run_id}] Brave error: {e}")
        return []
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


async def _ddg_search(query: str, k: int = 5, run_id: str = "-") -> List[dict]:
    url = f"https://duckduckgo.com/html/?q={query}"
    t0 = time.time()
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
            async with sess.get(url, timeout=25) as resp:
                html = await resp.text()
        dt = (time.time() - t0) * 1000
        _LOG.info(f"[assessor:{run_id}] DDG search ok in {dt:.1f} ms")
    except Exception as e:
        _LOG.error(f"[assessor:{run_id}] DDG error: {e}")
        return []
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


async def _tiered_search_impl(query: str, k: int, run_id: str) -> List[dict]:
    q = query.strip()
    if not q:
        return []
    res = await _tavily_search(q, k, run_id)
    if not res:
        res = await _brave_search(q, k, run_id)
    if not res:
        res = await _ddg_search(q, k, run_id)
    _LOG.info(f"[assessor:{run_id}] Search results: {len(res)}")
    return res


@tool
async def web_search(
    query: str, k: int = SEARCH_TOP_K, run_id: Optional[str] = None
) -> List[dict]:
    """
    Tiered web search. Returns a list of {title, url, snippet}.
    Order of providers: Tavily → Brave → DuckDuckGo.
    """
    rid = run_id or "-"
    return await _tiered_search_impl(query, k, rid)


@tool
async def web_fetch(
    url: str, max_bytes: int = FETCH_MAX_BYTES, run_id: Optional[str] = None
) -> dict:
    """
    Fetch a URL and return {url, title, text}. Truncates to max_bytes.
    """
    cap = min(max_bytes, ABSOLUTE_MAX_BYTES)
    t0 = time.time()
    try:
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
        dt = (time.time() - t0) * 1000
        _LOG.info(
            f"[assessor:{run_id or '-'}] Fetched {url[:80]}... in {dt:.1f} ms, {len(text)} chars"
        )
        return {"url": url, "title": title, "text": text[:cap]}
    except Exception as e:
        _LOG.error(f"[assessor:{run_id or '-'}] Fetch error for {url}: {e}")
        return {"url": url, "title": url, "text": ""}


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


def _build_graph(llm, tools: List[Tool], run_id: str, verbose: bool):
    g = StateGraph(AssessorState)

    async def decide(state: AssessorState):
        t0 = time.time()
        product = state["product"]
        state["do_research"] = _should_research(product)
        _LOG.info(
            f"[assessor:{run_id}] Node decide -> do_research={state['do_research']} ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def research(state: AssessorState):
        t0 = time.time()
        if not state.get("do_research"):
            state["evidence"] = []
            _LOG.info(
                f"[assessor:{run_id}] Node research skipped ({(time.time()-t0)*1000:.1f} ms)"
            )
            return state
        p = state["product"]
        seed_url = getattr(p, "url", "") or ""
        q = f'{getattr(p, "name", "")} {getattr(p, "category", "")} crypto token whitepaper roadmap security team site:{seed_url}'
        _LOG.info(f"[assessor:{run_id}] Searching: {q}")
        search_res = await web_search.ainvoke(
            {"query": q, "k": SEARCH_TOP_K, "run_id": run_id}
        )
        top = search_res[:FETCH_TOP_N] if isinstance(search_res, list) else []
        fetched = []
        for r in top:
            try:
                doc = await web_fetch.ainvoke({"url": r["url"], "run_id": run_id})
                fetched.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                        "text": doc.get("text", ""),
                    }
                )
            except Exception as e:
                _LOG.error(f"[assessor:{run_id}] Fetch pipeline error: {e}")
                continue
        state["evidence"] = fetched if fetched else top
        _LOG.info(
            f"[assessor:{run_id}] Node research -> evidence={len(state['evidence'])} ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def score(state: AssessorState):
        t0 = time.time()
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
        _LOG.info(f"[assessor:{run_id}] Node score -> invoking LLM")
        msg = await llm.ainvoke(prompt)
        state["raw_text"] = msg.content if isinstance(msg, AIMessage) else str(msg)
        _LOG.info(
            f"[assessor:{run_id}] Node score -> received {len(state['raw_text'] or '')} chars ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def validate(state: AssessorState):
        t0 = time.time()
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
        _LOG.info(
            f"[assessor:{run_id}] Node validate -> done ({(time.time()-t0)*1000:.1f} ms)"
        )
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
async def run_assessor(llm_big, product, *, verbose: bool = False) -> Dict[str, Any]:
    """
    Drop-in entry point:
        parsed = await run_assessor(llm_big=llm_big, product=product, verbose=True)
    """
    # Raise verbosity of aiohttp if needed
    if verbose and hasattr(_LOG, "setLevel"):
        try:
            _LOG.setLevel(logging.DEBUG)
            logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
        except Exception:
            pass

    run_id = uuid.uuid4().hex[:8]
    _LOG.info(
        f"[assessor:{run_id}] Start assessment for '{getattr(product,'name','?')}'"
    )

    # Optional LC tracing (enable by env)
    #   LANGCHAIN_TRACING_V2=true
    #   LANGCHAIN_API_KEY=...
    #   LANGCHAIN_PROJECT=CheckerChain
    callbacks = [AssessorCallbackHandler(run_id, verbose=verbose)]

    # Enable tool use on your big model
    llm = llm_big.bind_tools([web_search, web_fetch])

    graph = _build_graph(llm, [web_search, web_fetch], run_id, verbose)
    state = {"product": product}

    # Pass callbacks through RunnableConfig so both LLM and tools are instrumented
    cfg = RunnableConfig(callbacks=callbacks, tags=["assessor", run_id])

    t0 = time.time()
    result = await graph.ainvoke(state, config=cfg)
    dt = (time.time() - t0) * 1000
    _LOG.info(f"[assessor:{run_id}] Finished in {dt:.1f} ms")

    return result["final"]
