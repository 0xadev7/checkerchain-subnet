import os, re, urllib.parse, requests, time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.searx_search import SearxSearchWrapper
from langchain_community.tools.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults


import bittensor as bt

from checkerchain.utils.config import (
    REQUEST_TIMEOUT_SECS,
    SEARCH_RESULT_LIMIT,
    SCRAPE_PER_QUERY_LIMIT,
)

# ---------------------------
# Helpers
# ---------------------------

OFFICIAL_HINTS = (
    "docs.",
    "documentation",
    "whitepaper",
    "litepaper",
    "audit",
    "security",
    "github.com",
    "gitbook.io",
    "roadmap",
    "community",
    "discord.gg",
    "twitter.com",
    "x.com",
    "forum",
    "gov",
    "snapshot.org",
    "medium.com",
    "blog.",
)

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
    u = u.strip().split("#")[0]
    # remove tracking params
    parsed = urllib.parse.urlsplit(u)
    clean_qs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    qs = urllib.parse.urlencode(
        [(k, v) for k, v in clean_qs if not k.startswith(("utm_", "ref"))]
    )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, qs, ""))


def netloc(u: str) -> str:
    try:
        return urllib.parse.urlsplit(u).netloc.lower()
    except Exception:
        return ""


def is_pdf(u: str) -> bool:
    return urllib.parse.urlsplit(u).path.lower().endswith(".pdf")


def prefer_official(product_url: Optional[str], u: str) -> int:
    """Simple score: prefer official domain, docs/whitepaper/audits/github/roadmaps."""
    score = 0
    nl = netloc(u)
    if product_url and netloc(product_url).replace("www.", "") in nl.replace(
        "www.", ""
    ):
        score += 50
    # common official doc hosts
    if "gitbook.io" in nl or "docs." in nl:
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
    # deprioritize obvious noise
    if any(bad in nl for bad in ("pinterest.", "quora.", "fb.", "facebook.")):
        score -= 15
    return score


def unique_by_domain_and_url(urls: List[str]) -> List[str]:
    seen = set()
    seen_domain = set()
    result = []
    for u in urls:
        nu = normalize_url(u)
        if nu in seen:
            continue
        d = netloc(nu)
        # allow up to 2 per domain to avoid one site dominating
        key = (d, len([x for x in result if netloc(x) == d]) >= 2)
        if key[1]:
            continue
        seen.add(nu)
        result.append(nu)
    return result


# ---------------------------
# Free/Free-tier Search
# ---------------------------


def _search_tavily(query: str, k: int) -> List[str]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        tavily = TavilySearchAPIWrapper(
            tavily_api_key=api_key,
            max_results=k,
            include_answer=False,
            include_images=False,
        )
        # returns list of dicts with 'url'
        hits = tavily.results(query).get("results", [])
        return [h["url"] for h in hits if isinstance(h, dict) and h.get("url")]
    except Exception as e:
        bt.logging.warning("Tavily search failed", e)
        return []


def _search_searx(query: str, k: int) -> List[str]:
    host = os.getenv("SEARX_HOST")  # e.g. https://searxng.site or self-hosted
    if not host:
        return []
    try:
        searx = SearxSearchWrapper(
            searx_host=host, k=k, engines=None
        )  # let instance default
        results = searx.results(query)
        # typical structure: [{'title':..., 'link':..., 'snippet':...}, ...]
        return [r["link"] for r in results if isinstance(r, dict) and r.get("link")]
    except Exception as e:
        bt.logging.warning("SearxNG search failed", e)
        return []


def _search_ddg(query: str, k: int) -> List[str]:
    try:
        ddg = DuckDuckGoSearchResults(max_results=k)
        raw = ddg.run(query)
        return re.findall(r'https?://[^\s)"]+', raw)[:k]
    except Exception as e:
        bt.logging.warning("DuckDuckGo search failed", e)
        return []


def web_search(query: str, limit: int = SEARCH_RESULT_LIMIT) -> List[str]:
    """
    Free/Free-tier cascade: Tavily -> SearxNG -> DuckDuckGo
    """
    for fn in (_search_tavily, _search_searx, _search_ddg):
        urls = fn(query, limit)
        if urls:
            # normalize and dedupe lightly here
            urls = [normalize_url(u) for u in urls if u.startswith("http")]
            return unique_by_domain_and_url(urls)[:limit]
    return []


# ---------------------------
# Context dataset (adds Wikipedia as a free boost)
# ---------------------------


def fetch_product_dataset(name: str, url: Optional[str] = None) -> str:
    ctx = []

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

    # Wikipedia (free, often decent for quick background)
    try:
        wiki = WikipediaAPIWrapper(lang="en")
        page = wiki.run(name)  # returns top summary text
        if page and isinstance(page, str) and len(page) > 200:
            ctx.append(f"WIKIPEDIA: {page}")
    except Exception:
        pass

    # Fallback: official site scrape
    if not ctx and url:
        try:
            docs = WebBaseLoader(
                url, requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            ctx.extend([d.page_content for d in docs])
        except Exception:
            pass

    return "\n".join(ctx) if ctx else ""


# ---------------------------
# Web context: targeted queries + prioritization
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


def expand_queries(product_name: str, product_url: Optional[str]) -> List[str]:
    # Use domain without www for site-restricted precision when available
    base = product_name.strip()
    if product_url:
        dom = netloc(product_url).replace("www.", "")
        site_q = f"site:{dom}"
    else:
        site_q = base
    qs = []
    for tmpl in TARGET_QUERIES:
        qs.append(tmpl.format(k=site_q))
        if site_q != base:
            qs.append(tmpl.format(k=base))
    # de-dup while keeping order
    seen = set()
    uniq = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def fetch_web_context(
    product_name: str, product_url: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Search the web and scrape top pages (preferring official docs/audits/whitepaper/github/roadmap/community).
    Returns list of (url, page_text).
    """
    queries = expand_queries(product_name, product_url)

    # Collect and score URLs
    candidate_urls: List[str] = []
    seen = set()
    for q in queries:
        hits = web_search(q, limit=max(4, SEARCH_RESULT_LIMIT // 2))
        for u in hits:
            if u not in seen:
                seen.add(u)
                candidate_urls.append(u)
        if len(candidate_urls) >= max(SCRAPE_PER_QUERY_LIMIT * 2, 12):
            break

    if not candidate_urls:
        bt.logging.info("No URLs found for web context.")
        return []

    # Score and sort
    scored = sorted(
        candidate_urls,
        key=lambda u: prefer_official(product_url, u),
        reverse=True,
    )

    # Deduplicate by domain and finalize
    final_urls = unique_by_domain_and_url(scored)[:SCRAPE_PER_QUERY_LIMIT]

    bt.logging.info("Use these URLs for web context:\n" + "\n".join(final_urls))

    # Scrape
    pages: List[Tuple[str, str]] = []
    for u in final_urls:
        if is_pdf(u):
            # Optional: integrate UnstructuredURLLoader if you want PDF parsing.
            bt.logging.info(f"Skipping PDF for now (no PDF loader wired): {u}")
            continue
        try:
            docs = WebBaseLoader(
                u, requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            for d in docs:
                if d and getattr(d, "page_content", None):
                    pages.append((u, d.page_content))
        except Exception as e:
            bt.logging.warning(f"Error fetching pages from URL: {u} | {e}")
            continue

    return pages
