"""
Task 4: Prepare data for semantic search.

Splits raw Document objects into overlapping chunks sized for embedding.
Matches the "fixed size + overlap" strategy from the class notes
(CharacterTextSplitter-style chunking with overlap to avoid losing context
at chunk boundaries).
"""
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
    chunks = splitter.split_documents(docs)

    # Give every chunk a stable id so we can trace answers back to a source
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}-{i}"
    return chunks
