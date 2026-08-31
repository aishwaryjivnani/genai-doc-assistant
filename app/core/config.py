"""
Task 1: Project foundation - central configuration.
Everything is read from environment variables (.env) so no secrets or
paths are hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM - OpenAI Responses API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    # Embeddings (local model, no API key required)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # Retrieval
    # TOP_K is the number of final chunks sent to the LLM. Retrieve a larger
    # candidate set first so exact terms can compete with semantic matches.
    TOP_K: int = int(os.getenv("TOP_K", 5))
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", 8))
    # Chroma returns a distance (lower is better), not a relevance score.
    # Calibrate this against the application's own documents.
    MAX_DISTANCE: float = float(os.getenv("MAX_DISTANCE", 1.2))

    # Storage
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "data/chroma_db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    # Keep collections isolated when the embedding implementation changes so
    # vectors from the previous model are never mixed with new vectors.
    COLLECTION_NAME: str = "enterprise_docs_fastembed"
    CHROMA_ANONYMIZED_TELEMETRY: bool = (
        os.getenv("CHROMA_ANONYMIZED_TELEMETRY", "false").lower() == "true"
    )

    # Agent controls
    MAX_REASONING_RETRIES: int = int(os.getenv("MAX_REASONING_RETRIES", 2))

    # API server
    API_PORT: int = int(os.getenv("API_PORT", 8080))


settings = Settings()

os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
