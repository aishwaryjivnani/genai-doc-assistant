"""
Task 1: Project foundation - central configuration.
Everything is read from environment variables (.env) so no secrets or
paths are hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM - Google Gemini (free tier)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Embeddings (local model, no API key required)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", 4))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", 0.25))

    # Storage
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "data/chroma_db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    COLLECTION_NAME: str = "enterprise_docs"

    # Agent controls
    MAX_REASONING_RETRIES: int = int(os.getenv("MAX_REASONING_RETRIES", 2))

    # API server
    API_PORT: int = int(os.getenv("API_PORT", 8080))


settings = Settings()

os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
