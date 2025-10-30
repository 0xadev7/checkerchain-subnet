from __future__ import annotations
import os, logging

try:
    import bittensor as bt

    LOG = bt.logging
except Exception:
    LOG = logging.getLogger("assessor")
    if not LOG.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)

# ---------- Constants ----------
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

# API keys from env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()
