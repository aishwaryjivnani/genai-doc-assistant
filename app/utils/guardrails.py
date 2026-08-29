"""
Task 9: Reliability and safety controls.

Maps directly to the "Control:" checklist from the class notes:
  - Input validation (file type, file size, file content)
  - Error handling + logging
  - Monitor the model (left as a hook — see note below)
  - Result verification agent (implemented as the Validator agent in
    app/agents/graph.py)
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger("genai_doc_assistant")
logging.basicConfig(level=logging.INFO)

MAX_QUESTION_LEN = 2000
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml"}


def validate_question(question: str) -> Tuple[bool, str]:
    """Basic input validation before any retrieval/LLM call happens."""
    if not question or not question.strip():
        return False, "Question is empty."
    if len(question) > MAX_QUESTION_LEN:
        return False, f"Question is too long (max {MAX_QUESTION_LEN} characters)."
    return True, ""


def validate_upload(filename: str, size_bytes: int) -> Tuple[bool, str]:
    """
    File type + file size validation, per the "Input Validation" control
    in the class notes. File content validation (e.g. detecting corrupt
    or malformed files) happens inside the ingestion loaders themselves,
    where a parse failure is caught and logged.
    """
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB."
    return True, ""


def has_sufficient_context(results: List[Tuple[Document, float]]) -> bool:
    """
    Chroma's default distance metric: lower score = more similar.
    If even the best match is too far away, we don't have grounding for an
    answer, so the agent should say "I don't know" instead of guessing.
    """
    if not results:
        return False
    best_score = min(score for _, score in results)
    # Chroma returns L2 distances; typical sentence-embedding distances for
    # relevant content sit between 0–2. Use a flat upper bound of 2.0 so that
    # MIN_RELEVANCE_SCORE (0–1) scales it proportionally (0.25 → threshold 1.5
    # was too tight; now 2.0 is the baseline and MIN_RELEVANCE_SCORE trims it).
    threshold = 2.0 * (1 - settings.MIN_RELEVANCE_SCORE / 2)
    return best_score <= threshold


NO_CONTEXT_RESPONSE = (
    "I don't have enough relevant information in the uploaded documents to "
    "answer that confidently. Try rephrasing, or upload a document that "
    "covers this topic."
)


def strip_prompt_injection_markers(text: str) -> str:
    """
    Very lightweight defense: if retrieved document text contains phrases
    that look like they're trying to override instructions, neutralize them
    before they reach the LLM prompt.
    """
    suspicious_phrases = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard the system prompt",
        "you are now",
    ]
    cleaned = text
    for phrase in suspicious_phrases:
        if phrase in cleaned.lower():
            cleaned = cleaned.replace(phrase, "[filtered]")
    return cleaned


def log_event(event: str, **details) -> None:
    """Simple structured logging hook — the 'monitor the model' control."""
    logger.info("event=%s details=%s", event, details)
