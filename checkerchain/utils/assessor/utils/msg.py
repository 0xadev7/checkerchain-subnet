from typing import Any, List, Dict
from langchain_core.messages import AIMessage, BaseMessage


def extract_text_from_message(msg: BaseMessage | Any) -> str:
    """
    Robustly extract human-readable text from a LangChain message.
    Handles string, list-of-blocks, and various provider shapes.
    """
    # 1) Direct string content
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content

    # 2) List-of-blocks (OpenAI/Anthropic style)
    texts: List[str] = []
    if isinstance(content, list):
        for part in content:
            # plain strings inside list
            if isinstance(part, str):
                texts.append(part)
                continue
            # dict blocks
            if isinstance(part, dict):
                t = part.get("type")
                # OpenAI: {"type":"text","text": "..."} or {"type":"text","text":{"value":"..."}}
                if t == "text":
                    text_payload = part.get("text")
                    if isinstance(text_payload, dict) and "value" in text_payload:
                        texts.append(str(text_payload["value"]))
                    elif isinstance(text_payload, str):
                        texts.append(text_payload)
                    else:
                        # last resort stringify
                        texts.append(str(text_payload))
                # Ignore tool/assistant control blocks by default
                elif t in {"tool_use", "tool_result", "function_call", "input_text"}:
                    continue
                else:
                    # unknown block—stringify if useful
                    if "text" in part and isinstance(part["text"], str):
                        texts.append(part["text"])
                    else:
                        # avoid dumping huge JSON; skip
                        continue
            else:
                # unknown type—stringify safely
                try:
                    texts.append(str(part))
                except Exception:
                    pass
        if texts:
            return "\n".join([t for t in texts if t])

    # 3) Some providers stash text elsewhere
    ak = getattr(msg, "additional_kwargs", {}) or {}
    if isinstance(ak, dict):
        fc = ak.get("function_call") or ak.get("tool_calls")
        if isinstance(fc, dict) and "arguments" in fc:
            # not ideal as "text", but better than nothing
            return str(fc["arguments"])
        if isinstance(fc, list):
            # concat arguments strings if present
            args = []
            for c in fc:
                a = (
                    c.get("function", {}).get("arguments")
                    if isinstance(c, dict)
                    else None
                )
                if isinstance(a, str):
                    args.append(a)
            if args:
                return "\n".join(args)

    # 4) Final fallback
    return str(content if content is not None else msg)
