from __future__ import annotations
from langchain_core.callbacks import BaseCallbackHandler
from .config import LOG


class AssessorCallbackHandler(BaseCallbackHandler):
    def __init__(self, run_id: str, verbose: bool = False):
        self.run_id = run_id
        self.verbose = verbose

    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.verbose:
            model = (serialized or {}).get("id", "llm")
            LOG.info(
                f"[assessor:{self.run_id}] LLM start: {model} | prompts={len(prompts)}"
            )

    def on_llm_end(self, response, **kwargs):
        if self.verbose:
            usage = getattr(response, "llm_output", {}) or {}
            LOG.info(f"[assessor:{self.run_id}] LLM end: usage={usage}")

    def on_llm_error(self, error, **kwargs):
        LOG.error(f"[assessor:{self.run_id}] LLM error: {error}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.verbose:
            name = (serialized or {}).get("name", "tool")
            LOG.info(f"[assessor:{self.run_id}] Tool start: {name} | input={input_str}")

    def on_tool_end(self, output, **kwargs):
        if self.verbose:
            summary = str(output)
            if len(summary) > 500:
                summary = summary[:500] + "...[trunc]"
            LOG.info(f"[assessor:{self.run_id}] Tool end: {summary}")

    def on_tool_error(self, error, **kwargs):
        LOG.error(f"[assessor:{self.run_id}] Tool error: {error}")
