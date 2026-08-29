"""
Task 5 & 6: Vector-based knowledge store + intelligent retrieval.

Chroma DB, chosen from the options in the class notes (Chroma / FAISS /
Qdrant / Pinecone) because it runs locally with zero setup and persists
to disk — no server or cloud account needed for a personal project.

Flow (per class notes): generate embedding -> store embedding with
metadata -> index for similarity search -> query processing -> cosine
similarity search -> return top-N results.
"""
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.VECTOR_DB_DIR,
        )
    return _vectorstore


def add_chunks(chunks: List[Document]) -> int:
    """Embeds chunks and stores them (with metadata) in Chroma."""
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Cosine-similarity search. Returns (document, distance_score) pairs,
    most relevant first (lower distance = more similar).
    """
    store = get_vectorstore()
    k = k or settings.TOP_K
    return store.similarity_search_with_score(query, k=k)


def collection_count() -> int:
    store = get_vectorstore()
    return store._collection.count()


def reset_collection() -> None:
    """Wipes all stored chunks. Useful when re-processing a document set."""
    global _vectorstore
    store = get_vectorstore()
    store.delete_collection()
    _vectorstore = None
