from __future__ import annotations

import os
import re
import urllib.parse
import requests
from typing import List, Tuple, Optional
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults

import bittensor as bt

from checkerchain.utils.config import (
    REQUEST_TIMEOUT_SECS,
    SEARCH_RESULT_LIMIT,
    SCRAPE_PER_QUERY_LIMIT,
)

# ---------------------------
# URL / scoring helpers
# ---------------------------

AUDIT_KEYWORDS = (
    "audit",
    "security review",
    "trail of bits",
    "least authority",
    "sherlock",
    "code4rena",
    "immunefi",
    "quantstamp",
    "halborn",
)
COMMUNITY_KEYWORDS = ("discord", "telegram", "forum", "snapshot")


def normalize_url(u: str) -> str:
    """Strip fragments and tracking params for better dedupe."""
    try:
        u = u.strip().split("#")[0]

        parsed = urlparse(u)

        # If missing scheme, assume https
        if not parsed.scheme:
            url = "https://" + u
            parsed = urlparse(url)

        clean_qs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        qs = urllib.parse.urlencode(
            [(k, v) for k, v in clean_qs if not k.startswith(("utm_", "ref"))]
        )
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, qs, "")
        )
    except Exception:
        return u.strip()


def netloc(u: str) -> str:
    try:
        return urllib.parse.urlsplit(u).netloc.lower()
    except Exception:
        return ""


def is_pdf(u: str) -> bool:
    return urllib.parse.urlsplit(u).path.lower().endswith(".pdf")


def prefer_official(product_url: Optional[str], u: str) -> int:
    """Score URLs to prefer official domains, docs/gitbook, github, audits, whitepapers, roadmaps & community."""
    score = 0
    nl = netloc(u)
    if product_url:
        try:
            official = netloc(product_url).replace("www.", "")
            if official and official in nl.replace("www.", ""):
                score += 50
        except Exception:
            pass

    # docs & code
    if "gitbook.io" in nl or nl.startswith("docs.") or "docs." in nl:
        score += 30
    if "github.com" in nl:
        score += 28

    # keywords
    lu = u.lower()
    for kw in AUDIT_KEYWORDS:
        if kw in lu:
            score += 20
    for kw in ("whitepaper", "litepaper", "roadmap"):
        if kw in lu:
            score += 12
    for kw in COMMUNITY_KEYWORDS:
        if kw in lu:
            score += 6

    # mild penalty for noisy platforms
    if any(bad in nl for bad in ("pinterest.", "quora.", "facebook.", "fb.")):
        score -= 15

    return score


def unique_by_domain_and_url(urls: List[str]) -> List[str]:
    """Allow up to 2 results per domain; normalize URLs; dedupe exact URLs."""
    seen_urls = set()
    domain_counts: dict[str, int] = {}
    out: List[str] = []
    for u in urls:
        nu = normalize_url(u)
        if not nu or nu in seen_urls:
            continue
        d = netloc(nu)
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if domain_counts[d] > 2:  # cap per domain
            continue
        seen_urls.add(nu)
        out.append(nu)
    return out


# ---------------------------
# Search backends (HTTP; no LangChain wrappers to avoid import churn)
# ---------------------------


def _search_tavily_http(query: str, k: int) -> List[str]:
    """
    Tavily free-tier via REST. Set TAVILY_API_KEY in env to enable.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={"query": query, "num_results": k},
            timeout=min(REQUEST_TIMEOUT_SECS, 15),
        )
        if not resp.ok:
            bt.logging.warning(
                f"Tavily HTTP search non-200: {resp.status_code} | {resp.text[:200]}"
            )
            return []
        data = resp.json()
        return [
            r["url"]
            for r in data.get("results", [])
            if isinstance(r, dict) and r.get("url")
        ]
    except Exception as e:
        bt.logging.warning(f"Tavily HTTP search failed: {e}")
        return []


def _search_searx_http(query: str, k: int) -> List[str]:
    """
    SearxNG public/self-hosted instance via REST.
    Set SEARX_HOST (e.g., https://searxng.site) to enable.
    """
    base = os.getenv("SEARX_HOST")
    if not base:
        return []
    try:
        resp = requests.get(
            f"{base.rstrip('/')}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 1,
                "categories": "general",
            },
            timeout=min(REQUEST_TIMEOUT_SECS, 15),
        )
        if not resp.ok:
            bt.logging.warning(
                f"SearxNG HTTP search non-200: {resp.status_code} | {resp.text[:200]}"
            )
            return []
        data = resp.json()
        results = data.get("results", [])
        urls = [r.get("url") or r.get("link") for r in results if isinstance(r, dict)]
        return [u for u in urls if u and u.startswith("http")][:k]
    except Exception as e:
        bt.logging.warning(f"SearxNG HTTP search failed: {e}")
        return []


def _search_ddg_tool(query: str, k: int) -> List[str]:
    """
    DuckDuckGo via LangChain tool (no key).
    """
    try:
        ddg = DuckDuckGoSearchResults(max_results=k)
        raw = ddg.run(query)
        urls = re.findall(r'https?://[^\s)"]+', raw)
        return urls[:k]
    except Exception as e:
        bt.logging.warning(f"DuckDuckGo tool failed: {e}")
        return []


def web_search(query: str, limit: int = SEARCH_RESULT_LIMIT) -> List[str]:
    """
    Free/Free-tier cascade: Tavily (HTTP) -> SearxNG (HTTP) -> DuckDuckGo tool.
    Returns normalized, lightly deduped URLs.
    """
    for fn in (_search_tavily_http, _search_searx_http, _search_ddg_tool):
        urls = fn(query, limit)
        if urls:
            urls = [normalize_url(u) for u in urls if u.startswith("http")]
            return unique_by_domain_and_url(urls)[:limit]
    return []


# ---------------------------
# Dataset enrichment (CoinGecko, Messari, DefiLlama, Wikipedia)
# ---------------------------


def fetch_product_dataset(name: str, url: Optional[str] = None) -> str:
    """
    Returns a joined string of background context about the project from free sources.
    """
    ctx: List[str] = []

    # Scrape Product URL
    if url:
        try:
            docs = WebBaseLoader(
                normalize_url(url), requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            ctx.extend(
                [d.page_content for d in docs if getattr(d, "page_content", None)]
            )
        except Exception as e:
            bt.logging.warning(f"Product URL scrape failed for {url}: {e}")

    # CoinGecko
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/search?query={name}",
            timeout=REQUEST_TIMEOUT_SECS,
        )
        if r.ok and r.json().get("coins"):
            coin_id = r.json()["coins"][0]["id"]
            d = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                timeout=REQUEST_TIMEOUT_SECS,
            ).json()
            summary = (d.get("description", {}) or {}).get("en", "")
            if summary:
                ctx.append(f"COINGECKO SUMMARY: {summary}")
    except Exception:
        pass

    # Messari
    try:
        r = requests.get(
            f"https://data.messari.io/api/v2/assets/{name.lower()}/profile",
            timeout=REQUEST_TIMEOUT_SECS,
        )
        if r.ok:
            prof = (r.json().get("data", {}) or {}).get("profile", {})
            summary = ((prof.get("general", {}) or {}).get("overview", {}) or {}).get(
                "project_details", ""
            )
            if summary:
                ctx.append(f"MESSARI OVERVIEW: {summary}")
    except Exception:
        pass

    # DefiLlama
    try:
        r = requests.get("https://api.llama.fi/protocols", timeout=REQUEST_TIMEOUT_SECS)
        if r.ok:
            protocols = r.json()
            m = next(
                (
                    p
                    for p in protocols
                    if name.lower() in (p.get("name", "") or "").lower()
                ),
                None,
            )
            if m and m.get("description"):
                ctx.append(f"DEFI LLAMA: {m['description']}")
    except Exception:
        pass

    return "\n".join(ctx) if ctx else ""


# ---------------------------
# Web context collector
# ---------------------------

TARGET_QUERIES = [
    "{k} official site",
    "{k} docs OR documentation",
    "{k} whitepaper OR litepaper filetype:pdf",
    "{k} audit OR security review",
    "{k} github",
    "{k} roadmap",
    "{k} community OR forum OR discord OR telegram",
]


def _expand_queries(product_name: str, product_url: Optional[str]) -> List[str]:
    base = product_name.strip()
    if product_url:
        site_q = f"site:{product_url}"
    else:
        site_q = base

    qs: List[str] = []
    for tmpl in TARGET_QUERIES:
        qs.append(tmpl.format(k=base))
        if site_q != base:
            qs.append(tmpl.format(k=site_q))

    # de-dup preserving order
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def fetch_web_context(
    product_name: str, product_url: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Search the web and scrape top pages (preferring official docs/audits/whitepaper/github/roadmap/community).
    Returns list of (url, page_text).
    """
    queries = _expand_queries(product_name, product_url)

    # Collect candidates across queries (stop early on enough hits)
    candidate_urls: List[str] = []
    seen = set()
    per_query_limit = max(4, SEARCH_RESULT_LIMIT // 2)
    cap = max(SCRAPE_PER_QUERY_LIMIT * 2, 12)

    for q in queries:
        hits = web_search(q, limit=per_query_limit)
        for u in hits:
            if u not in seen:
                seen.add(u)
                candidate_urls.append(u)
        if len(candidate_urls) >= cap:
            break

    if not candidate_urls:
        bt.logging.info("No URLs found for web context.")
        return []

    # Score → sort → dedupe per domain → cap
    scored = sorted(
        candidate_urls, key=lambda u: prefer_official(product_url, u), reverse=True
    )
    final_urls = unique_by_domain_and_url(scored)[:SCRAPE_PER_QUERY_LIMIT]

    bt.logging.info("Use these URLs for web context:\n" + "\n".join(final_urls))

    # Scrape pages (skip PDFs unless you wire a PDF loader)
    pages: List[Tuple[str, str]] = []
    for u in final_urls:
        if is_pdf(u):
            bt.logging.info(f"Skipping PDF (no PDF loader wired): {u}")
            continue
        try:
            docs = WebBaseLoader(
                u, requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            for d in docs:
                if d and getattr(d, "page_content", None):
                    pages.append((u, d.page_content))
        except Exception as e:
            bt.logging.warning(f"Error fetching URL {u}: {e}")
            continue

    return pages
