import asyncio
import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from checkerchain.utils.config import (
    DB_PATH,
    EMBED_MODEL,
    COLLECTION_NAME,
    OPENAI_API_KEY,
)


def get_vector_store():
    """
    Create/load a persistent Chroma collection using a PersistentClient.
    No explicit .persist() is needed; persistence is handled by the client.
    """
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    vs = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vs


def add_to_vector_store(context: str):
    """
    Add text to the persistent vector store.
    """
    if not context or not context.strip():
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    docs = splitter.create_documents([context])
    vs = get_vector_store()
    vs.add_documents(docs)


async def retrieve_context(query: str, k: int = 4) -> str:
    """
    Retrieve top-k relevant docs; supports new Runnable retrievers.
    """
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    try:
        docs = await retriever.ainvoke(query)  # New API
    except AttributeError:
        # Fallback to sync call in a thread for older versions
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, retriever.get_relevant_documents, query)
    contents = [
        d.page_content for d in (docs or []) if getattr(d, "page_content", None)
    ]
    return "\n".join(contents)
