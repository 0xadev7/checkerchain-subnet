from __future__ import annotations
import time, re, aiohttp
from typing import Optional
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from ..config import LOG, USER_AGENT, FETCH_MAX_BYTES, ABSOLUTE_MAX_BYTES


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
        LOG.info(
            f"[assessor:{run_id or '-'}] Fetched {url[:80]}... in {(time.time()-t0)*1000:.1f} ms, {len(text)} chars"
        )
        return {"url": url, "title": title, "text": text[:cap]}
    except Exception as e:
        LOG.error(f"[assessor:{run_id or '-'}] Fetch error for {url}: {e}")
        return {"url": url, "title": url, "text": ""}
