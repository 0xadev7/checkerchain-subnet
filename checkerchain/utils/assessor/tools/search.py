from __future__ import annotations
import time, aiohttp
from typing import List, Optional
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from ..config import LOG, USER_AGENT, SEARCH_TOP_K, TAVILY_API_KEY, BRAVE_API_KEY


async def _tavily_search(query: str, k: int, run_id: str) -> List[dict]:
    if not TAVILY_API_KEY:
        return []
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": k}
    t0 = time.time()
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
            async with sess.post(url, json=payload, timeout=25) as resp:
                if resp.status != 200:
                    LOG.warning(f"[assessor:{run_id}] Tavily HTTP {resp.status}")
                    return []
                data = await resp.json()
        LOG.info(
            f"[assessor:{run_id}] Tavily search ok in {(time.time()-t0)*1000:.1f} ms"
        )
    except Exception as e:
        LOG.error(f"[assessor:{run_id}] Tavily error: {e}")
        return []
    res = []
    for r in (data.get("results") or [])[:k]:
        res.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or "",
            }
        )
    return [r for r in res if r["url"]]


async def _brave_search(query: str, k: int, run_id: str) -> List[dict]:
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
                    LOG.warning(f"[assessor:{run_id}] Brave HTTP {resp.status}")
                    return []
                data = await resp.json()
        LOG.info(
            f"[assessor:{run_id}] Brave search ok in {(time.time()-t0)*1000:.1f} ms"
        )
    except Exception as e:
        LOG.error(f"[assessor:{run_id}] Brave error: {e}")
        return []
    out = []
    for r in (data.get("web") or {}).get("results", [])[:k]:
        out.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("description") or "",
            }
        )
    return [r for r in out if r["url"]]


async def _ddg_search(query: str, k: int, run_id: str) -> List[dict]:
    url = f"https://duckduckgo.com/html/?q={query}"
    t0 = time.time()
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as sess:
            async with sess.get(url, timeout=25) as resp:
                html = await resp.text()
        LOG.info(f"[assessor:{run_id}] DDG search ok in {(time.time()-t0)*1000:.1f} ms")
    except Exception as e:
        LOG.error(f"[assessor:{run_id}] DDG error: {e}")
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
    q = (query or "").strip()
    if not q:
        return []
    res = await _tavily_search(q, k, run_id)
    if not res:
        res = await _brave_search(q, k, run_id)
    if not res:
        res = await _ddg_search(q, k, run_id)
    LOG.info(f"[assessor:{run_id}] Search results: {len(res)}")
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
