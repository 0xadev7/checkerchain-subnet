import re, hashlib
from typing import List, Dict, Any
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def num_tokens(s: str) -> int:
    return len(ENC.encode(s or ""))


def canonicalize_url(u: str) -> str:
    return re.sub(r"[#?].*$", "", u.rstrip("/").lower())


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    # naive chunker; swap in recursive splitter if you have one
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for d in docs:
        url = canonicalize_url(d.get("url", ""))
        h = (url, content_hash(d.get("text", "")[:8000]))
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out
