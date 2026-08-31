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

from chromadb.config import Settings as ChromaSettings
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


def warmup_vectorstore() -> None:
    """Load the embedding model and run one inference before serving traffic."""
    get_vectorstore()
    get_embeddings().embed_query("startup embedding warmup")


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.VECTOR_DB_DIR,
            client_settings=ChromaSettings(
                is_persistent=True,
                persist_directory=settings.VECTOR_DB_DIR,
                anonymized_telemetry=settings.CHROMA_ANONYMIZED_TELEMETRY,
                chroma_product_telemetry_impl="app.utils.chroma_telemetry.NoOpTelemetry",
                chroma_telemetry_impl="app.utils.chroma_telemetry.NoOpTelemetry",
            ),
        )
    return _vectorstore


def add_chunks(chunks: List[Document]) -> int:
    """Embeds chunks and stores them (with metadata) in Chroma."""
    if not chunks:
        return 0
    store = get_vectorstore()
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def replace_source_chunks(source: str, chunks: List[Document]) -> int:
    """Replace one source file without leaving stale or duplicate chunks."""
    store = get_vectorstore()
    # langchain-chroma's high-level delete() currently forwards only ids;
    # use the underlying Chroma collection for metadata-based deletion.
    existing = store._collection.get(where={"source": source})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        store._collection.delete(ids=existing_ids)
    return add_chunks(chunks)


def similarity_search(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Returns (document, distance_score) pairs, most relevant first
    (lower distance = more similar). The default Chroma collection metric is
    distance-based; callers must not treat this value as a percentage score.
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
