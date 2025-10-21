import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

MAP_PROMPT = """You extract FACTS needed to assess a DeFi/crypto product.
For each chunk, return a JSON array of objects with:
- "claim": atomic fact
- "axis": one of ["project","userbase","utility","security","team","tokenomics","marketing","roadmap","clarity","partnerships","other"]
- "evidence": {"source_id": "<id>", "snippet": "<verbatim-or-quoted-phrase>"}
Only include verifiable claims present in the chunk. No opinions.
Return JSON ONLY.
"""

REDUCE_PROMPT = """Merge the following JSON arrays of facts.
- Remove duplicates/near-duplicates.
- Keep the strongest/most recent evidence when conflicting.
- Output up to {max_facts} facts per axis.
Return one JSON object: {"facts":[...]} with the same schema per item.
Return JSON ONLY.
"""


async def map_extract(llm, chunk_text: str, source_id: str) -> List[Dict[str, Any]]:
    msg = f"{MAP_PROMPT}\n\n<chunk source_id='{source_id}'>\n{chunk_text}\n</chunk>"
    res = await llm.ainvoke(
        [
            SystemMessage(content="You are a precise fact extractor."),
            HumanMessage(content=msg),
        ]
    )
    return json.loads(res.content)


async def reduce_facts(
    llm, arrays: List[List[Dict[str, Any]]], max_facts: int = 8
) -> Dict[str, Any]:
    combined = json.dumps([item for arr in arrays for item in arr])[:200000]  # safety
    res = await llm.ainvoke(
        [
            SystemMessage(content="You are a meticulous merger and deduper."),
            HumanMessage(
                content=REDUCE_PROMPT.format(max_facts=max_facts) + "\n\n" + combined
            ),
        ]
    )
    return json.loads(res.content)
