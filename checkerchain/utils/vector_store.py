import asyncio
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from checkerchain.utils.config import DB_PATH, EMBED_MODEL, OPENAI_API_KEY


def get_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)


def add_to_vector_store(context: str):
    if not context.strip():
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    docs = splitter.create_documents([context])
    vs = get_vector_store()
    vs.add_documents(docs)


async def retrieve_context(query: str, k: int = 4) -> str:
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
