import re, requests
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

import bittensor as bt

from checkerchain.utils.config import (
    SERPAPI_API_KEY,
    SERPER_API_KEY,
    REQUEST_TIMEOUT_SECS,
    SEARCH_RESULT_LIMIT,
    SCRAPE_PER_QUERY_LIMIT,
)


def fetch_product_dataset(name: str, url: str | None = None) -> str:
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
                (p for p in protocols if name.lower() in p.get("name", "").lower()),
                None,
            )
            if m and m.get("description"):
                ctx.append(f"DEFI LLAMA: {m['description']}")
    except Exception:
        pass

    # Fallback scrape of official site
    if not ctx and url:
        try:
            docs = WebBaseLoader(
                url, requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            ctx.extend([d.page_content for d in docs])
        except Exception:
            pass

    return "\n".join(ctx) if ctx else ""


def fetch_web_context(product_name: str, product_url: str | None = None) -> List:
    """
    Search the web and scrape top pages (preferring official docs).
    """
    queries = [
        f"{product_name} | {product_url} - official site",
        f"{product_name} | {product_url} - audit report",
        f"{product_name} | {product_url} - whitepaper tokenomics",
        f"{product_name} | {product_url} - team & community",
        f"{product_name} | {product_url} - roadmap",
    ]
    if product_url:
        queries.insert(0, f"site:{product_url} {product_name}")

    seen = set()
    urls: List[str] = []
    for q in queries:
        for u in web_search(q, limit=SEARCH_RESULT_LIMIT):
            valid_url = u.replace(",", "")
            if valid_url not in seen:
                seen.add(valid_url)
                urls.append(valid_url)
            if len(urls) >= SCRAPE_PER_QUERY_LIMIT:
                break
        if len(urls) >= SCRAPE_PER_QUERY_LIMIT:
            break

    joined_urls = "\n".join(urls)
    bt.logging.info(f"Use these URLs for web context:\n {joined_urls}")

    pages = []
    for u in urls[:SCRAPE_PER_QUERY_LIMIT]:
        try:
            docs = WebBaseLoader(
                u, requests_kwargs={"timeout": REQUEST_TIMEOUT_SECS}
            ).load()
            for d in docs:
                pages.append((u, d.page_content))
        except Exception as e:
            bt.logging.warning(f"Error fetching pages from URL: {u}", e)
            continue

    return pages


def web_search(query: str, limit: int = SEARCH_RESULT_LIMIT) -> List[str]:
    """
    Returns a list of result URLs using best-available provider:
    Serper -> SerpAPI -> Bing -> Brave (custom) -> DuckDuckGo.
    """
    # 1) Serper (Google Serper)
    if SERPER_API_KEY:
        try:
            serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY, k=limit)
            results = serper.results(query)
            urls = [i.get("link") for i in results.get("organic", []) if i.get("link")]
            if urls:
                return urls[:limit]
        except Exception as e:
            bt.logging.warning("Error with Google Serper API", e)
            pass

    # 2) SerpAPI
    if SERPAPI_API_KEY:
        try:
            serp = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
            results = serp.results(query)
            # results structure may vary; handle both dict/list
            urls = []
            if isinstance(results, dict) and "organic_results" in results:
                urls = [
                    r.get("link") for r in results["organic_results"] if r.get("link")
                ]
            elif isinstance(results, list):
                urls = [
                    str(r.get("link") or r.get("url"))
                    for r in results
                    if isinstance(r, dict)
                ]
            if urls:
                return urls[:limit]
        except Exception as e:
            bt.logging.warning("Error with Serp API", e)
            pass

    # 3) DuckDuckGo (no API key)
    try:
        ddg_tool = DuckDuckGoSearchResults(max_results=limit)
        # returns a JSON-ish string; parse links heuristically
        raw = ddg_tool.run(query)
        # very light URL extraction
        urls = re.findall(r'https?://[^\s)"]+', raw)
        if urls:
            return urls[:limit]
    except Exception as e:
        bt.logging.warning("Error with DuckDuckGo API", e)
        pass

    return []
