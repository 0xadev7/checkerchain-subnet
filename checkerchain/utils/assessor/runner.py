from __future__ import annotations
import uuid, time, logging
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from .config import LOG
from .callbacks import AssessorCallbackHandler
from .graph import build_graph


async def run_assessor(llm_big, product, *, verbose: bool = False) -> Dict[str, Any]:
    """
    Drop-in entry point:
        parsed = await run_assessor(llm_big=llm_big, product=product, verbose=True)
    """
    if verbose and hasattr(LOG, "setLevel"):
        try:
            LOG.setLevel(logging.DEBUG)
            logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
        except Exception:
            pass

    run_id = uuid.uuid4().hex[:8]
    LOG.info(
        f"[assessor:{run_id}] Start assessment for '{getattr(product,'name','?')}'"
    )

    callbacks = [AssessorCallbackHandler(run_id, verbose=verbose)]
    llm = llm_big.bind_tools([])

    graph = build_graph(llm, [], run_id, verbose)
    state = {"product": product}

    cfg = RunnableConfig(callbacks=callbacks, tags=["assessor", run_id])

    t0 = time.time()
    print(state, cfg, graph)
    result = await graph.ainvoke(state, config=cfg)
    LOG.info(f"[assessor:{run_id}] Finished in {(time.time()-t0)*1000:.1f} ms")

    return result["final"]
