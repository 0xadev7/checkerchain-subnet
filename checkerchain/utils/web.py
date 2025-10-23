from __future__ import annotations

import os
import re
import urllib.parse
import requests
from typing import List, Tuple, Optional, Dict
from urllib.parse import (
    urlparse,
    urlencode,
    urlsplit,
    urlunsplit,
    parse_qs,
    quote_plus,
    unquote,
)

from langchain_community.document_loaders import WebBaseLoader
import bittensor as bt

from checkerchain.database.mongo import (
    get_cached_urls,
    upsert_cached_urls,
)
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
    """Strip fragments and tracking params for better dedupe. If missing scheme, assume https."""
    try:
        u = u.strip().split("#")[0]
        parsed = urlparse(u)

        # If missing scheme, assume https
        if not parsed.scheme:
            url = "https://" + u
            parsed = urlparse(url)

        clean_qs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        qs = urllib.parse.urlencode(
            [
                (k, v)
                for (k, v) in clean_qs
                if not k.startswith(("utm_", "ref", "fbclid"))
            ]
        )
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, qs, ""))
    except Exception:
        return u.strip()


def netloc(u: str) -> str:
    try:
        return urlsplit(u).netloc.lower()
    except Exception:
        return ""


def is_pdf(u: str) -> bool:
    return urlsplit(u).path.lower().endswith(".pdf")


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
    domain_counts: Dict[str, int] = {}
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
# One-shot query builder
# ---------------------------


def _build_one_shot_query(product_name: str, product_url: Optional[str]) -> str:
    """
    Build a single, rich query for all high-signal resources in one go.
    """
    base = product_name.strip()
    site_hint = f" ({product_url})" if product_url else ""
    return f"{base}{site_hint}: official website, security audits, whitepaper or litepaper, roadmap, community links, docs, github"


# ---------------------------
# Primary: Tavily (optional)
# ---------------------------


def _tavily_one_shot_search(
    product_name: str, product_url: Optional[str], k: int
) -> List[str]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        bt.logging.info("TAVILY_API_KEY not set; skipping Tavily search.")
        return []

    query = _build_one_shot_query(product_name, product_url)

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "num_results": max(k, SEARCH_RESULT_LIMIT),
                "include_images": False,
                "include_answer": False,
                "search_depth": "advanced",
            },
            timeout=min(REQUEST_TIMEOUT_SECS, 20),
        )
        if not resp.ok:
            bt.logging.warning(
                f"Tavily HTTP search non-200: {resp.status_code} | {resp.text[:200]}"
            )
            return []
        data = resp.json()
        urls = [
            r.get("url")
            for r in data.get("results", [])
            if isinstance(r, dict)
            and isinstance(r.get("url"), str)
            and r["url"].startswith("http")
        ]
        return unique_by_domain_and_url([normalize_url(u) for u in urls])[:k]
    except Exception as e:
        bt.logging.warning(f"Tavily HTTP search failed: {e}")
        return []


# ---------------------------
# Fallback 1: Brave Search API
# ---------------------------


def _brave_search(product_name: str, product_url: Optional[str], k: int) -> List[str]:
    """
    Brave Web Search API.
    Env: BRAVE_API_KEY
    Docs: https://api.search.brave.com/app/documentation/web-search
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        bt.logging.info("BRAVE_API_KEY not set; skipping Brave search.")
        return []

    query = _build_one_shot_query(product_name, product_url)
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
                "User-Agent": "checkerchain/1.0",
            },
            params={
                "q": query,
                "count": max(k, SEARCH_RESULT_LIMIT),
                "country": "us",
                "safesearch": "moderate",
            },
            timeout=min(REQUEST_TIMEOUT_SECS, 20),
        )
        if not resp.ok:
            bt.logging.warning(
                f"Brave HTTP search non-200: {resp.status_code} | {resp.text[:200]}"
            )
            return []
        data = resp.json() or {}
        results = (data.get("web") or {}).get("results") or []
        urls = [
            r.get("url")
            for r in results
            if isinstance(r, dict)
            and isinstance(r.get("url"), str)
            and r["url"].startswith("http")
        ]
        return unique_by_domain_and_url([normalize_url(u) for u in urls])[:k]
    except Exception as e:
        bt.logging.warning(f"Brave HTTP search failed: {e}")
        return []


# ---------------------------
# Fallback 2: DuckDuckGo (HTML scrape)
# ---------------------------

_DDG_RESULT_HREF = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', re.I)


def _ddg_unwrap_uddg(u: str) -> str:
    """
    DDG HTML links often look like: /l/?kh=-1&uddg=<percent-encoded URL>
    This extracts and returns the underlying URL.
    """
    try:
        if u.startswith("/l/") or u.startswith("https://duckduckgo.com/l/"):
            qs = parse_qs(urlsplit(u).query)
            uddg = qs.get("uddg", [None])[0]
            if uddg:
                return unquote(uddg)
        return u
    except Exception:
        return u


def _duckduckgo_html_search(
    product_name: str, product_url: Optional[str], k: int
) -> List[str]:
    """
    Scrape DuckDuckGo's HTML results (no API key).
    NOTE: This is best-effort and may change with DDG markup.
    """
    query = _build_one_shot_query(product_name, product_url)
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}&kl=us-en"

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; checkerchain/1.0; +https://example.com/bot)",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=min(REQUEST_TIMEOUT_SECS, 20),
        )
        if not resp.ok:
            bt.logging.warning(
                f"DuckDuckGo HTML non-200: {resp.status_code} | {resp.text[:200]}"
            )
            return []
        html = resp.text or ""
        raw_hrefs = _DDG_RESULT_HREF.findall(html)
        # unwrap uddg and keep http(s) only
        urls = []
        for href in raw_hrefs:
            real = _ddg_unwrap_uddg(href)
            if isinstance(real, str) and real.startswith("http"):
                urls.append(real)
        return unique_by_domain_and_url([normalize_url(u) for u in urls])[:k]
    except Exception as e:
        bt.logging.warning(f"DuckDuckGo HTML search failed: {e}")
        return []


# ---------------------------
# Dataset enrichment (CoinGecko, Messari, DefiLlama, Wikipedia-like sources)
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
# Web context collector (Tavily + Brave + DuckDuckGo fallbacks)
# ---------------------------


def fetch_web_context(
    product_name: str,
    product_url: Optional[str] = None,
    *,
    product_id: Optional[str] = None,
    review_cycle: int = 0,
    force_refresh: bool = False,
) -> List[Tuple[str, str]]:
    """
    Retrieve candidate links via Tavily, falling back to Brave then DuckDuckGo.
    Score + sort + dedupe per domain, then scrape.
    Returns list of (url, page_text).
    """
    # 1) Try cache first (only if product_id is provided)
    cached_urls: List[str] = []
    used_cache = False
    if product_id and not force_refresh:
        got = get_cached_urls(product_id, review_cycle)
        if got:
            cached_urls = got
            used_cache = True
            bt.logging.info(
                f"[web-cache] Using {len(cached_urls)} cached URLs for productId={product_id}, rc={review_cycle}"
            )

    # 2) If no cache, query providers in order
    if used_cache:
        candidate_urls = cached_urls
        provider_chain = ["cache"]
    else:
        k = max(SCRAPE_PER_QUERY_LIMIT * 2, 10)
        urls_tavily = _tavily_one_shot_search(product_name, product_url, k=k)
        urls_brave = (
            _brave_search(product_name, product_url, k=k)
            if len(urls_tavily) < k
            else []
        )
        urls_ddg = (
            _duckduckgo_html_search(product_name, product_url, k=k)
            if (len(urls_tavily) + len(urls_brave)) < k
            else []
        )

        # Merge in priority order
        candidate_urls = unique_by_domain_and_url(
            [*urls_tavily, *urls_brave, *urls_ddg]
        )
        provider_chain = [
            p
            for p, arr in (
                ("tavily", urls_tavily),
                ("brave", urls_brave),
                ("duckduckgo", urls_ddg),
            )
            if arr
        ]

        # Save to cache if possible
        if product_id and candidate_urls:
            try:
                upsert_cached_urls(
                    product_id=product_id,
                    review_cycle=review_cycle,
                    urls=candidate_urls,
                    source="+".join(provider_chain),
                    meta={
                        "requested": k,
                        "received": len(candidate_urls),
                        "productName": product_name,
                        "hasProductUrl": bool(product_url),
                    },
                )
                bt.logging.info(
                    f"[web-cache] Cached {len(candidate_urls)} URLs via {provider_chain} for productId={product_id}, rc={review_cycle}"
                )
            except Exception as e:
                bt.logging.warning(f"[web-cache] upsert failed: {e}")

    if not candidate_urls:
        bt.logging.info("No URLs found for web context.")
        return []

    # Score → sort → dedupe per domain → cap for scraping
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
