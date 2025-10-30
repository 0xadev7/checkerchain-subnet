from __future__ import annotations
import time
from typing import TypedDict, Any, Dict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from pydantic import ValidationError

from .config import LOG, SEARCH_TOP_K, FETCH_TOP_N, METRICS
from .prompts import SCORING_PROMPT
from .heuristics import should_research, format_evidence
from .utils.json_utils import parse_or_repair_json, coerce_with_defaults
from .models import AssessmentModel


class AssessorState(TypedDict, total=False):
    product: Any
    do_research: bool
    evidence: List[Dict[str, Any]]
    raw_text: str
    final: Dict[str, Any]


def build_graph(llm, tools: List[Tool], run_id: str, verbose: bool):
    from .tools import web_search, web_fetch

    g = StateGraph(AssessorState)

    async def decide(state: AssessorState):
        if "product" not in state:
            raise ValueError("State missing 'product' at entry.")

        t0 = time.time()
        product = state["product"]
        state["do_research"] = should_research(product)
        LOG.info(
            f"[assessor:{run_id}] Node decide -> do_research={state['do_research']} ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def research(state: AssessorState):
        t0 = time.time()
        if not state.get("do_research"):
            state["evidence"] = []
            LOG.info(
                f"[assessor:{run_id}] Node research skipped ({(time.time()-t0)*1000:.1f} ms)"
            )
            return state
        p = state.get("product")
        seed_url = getattr(p, "url", "") or ""
        q = f'{getattr(p, "name", "")} {getattr(p, "category", "")} crypto token whitepaper roadmap security team site:{seed_url}'
        LOG.info(f"[assessor:{run_id}] Searching: {q}")
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
                LOG.error(f"[assessor:{run_id}] Fetch pipeline error: {e}")
                continue
        state["evidence"] = fetched if fetched else top
        LOG.info(
            f"[assessor:{run_id}] Node research -> evidence={len(state['evidence'])} ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def grade(state: AssessorState):
        def _esc_braces(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        t0 = time.time()
        p = state.get("product")
        evidence = format_evidence(state.get("evidence", []))
        prompt = SCORING_PROMPT.format_messages(
            name=_esc_braces(getattr(p, "name", "")),
            description=_esc_braces(getattr(p, "description", "")),
            url=_esc_braces(getattr(p, "url", "")),
            category=getattr(getattr(p, "category", {}), "name", ""),
            evidence=_esc_braces(evidence),
            metrics=_esc_braces(", ".join(METRICS)),
        )
        LOG.info(f"[assessor:{run_id}] Node score -> invoking LLM")
        print("###", prompt)
        msg = await llm.ainvoke(prompt)
        print("***", msg)
        state["raw_text"] = msg.content if isinstance(msg, AIMessage) else str(msg)
        LOG.info(
            f"[assessor:{run_id}] Node score -> received {len(state['raw_text'] or '')} chars ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    async def validate(state: AssessorState):
        t0 = time.time()
        txt = state.get("raw_text", "")
        obj = parse_or_repair_json(txt)
        coerced = coerce_with_defaults(obj)
        try:
            parsed = AssessmentModel.model_validate(coerced).model_dump()
        except ValidationError:
            parsed = AssessmentModel.model_validate(
                coerce_with_defaults(coerced)
            ).model_dump()
        state["final"] = parsed
        LOG.info(
            f"[assessor:{run_id}] Node validate -> done ({(time.time()-t0)*1000:.1f} ms)"
        )
        return state

    g.add_node("decide", decide)
    g.add_node("research", research)
    g.add_node("grade", grade)
    g.add_node("validate", validate)

    g.set_entry_point("decide")
    g.add_edge("decide", "research")
    g.add_edge("research", "grade")
    g.add_edge("grade", "validate")
    g.add_edge("validate", END)
    return g.compile()
