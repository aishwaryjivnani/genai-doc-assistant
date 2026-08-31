"""
Task 4: Prepare data for semantic search.

Splits raw Document objects into overlapping chunks sized for embedding.
Matches the "fixed size + overlap" strategy from the class notes
(CharacterTextSplitter-style chunking with overlap to avoid losing context
at chunk boundaries).
"""
import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        # Structured records must not be cut in the middle of a row/object.
        # The ingestion layer already groups them into retrieval-sized units.
        if doc.metadata.get("structured"):
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    # Give every chunk a content-stable id. Chroma receives this id explicitly
    # during indexing, which makes re-uploading a source deterministic.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_count"] = len(chunks)
        identity = "|".join(
            str(chunk.metadata.get(key, ""))
            for key in ("source", "page", "sheet", "row_start", "record_start")
        )
        digest = hashlib.sha1(
            f"{identity}|{i}|{chunk.page_content}".encode("utf-8")
        ).hexdigest()[:16]
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}-{digest}"
    return chunks
