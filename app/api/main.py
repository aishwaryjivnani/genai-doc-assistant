"""
Task 2: User interaction layer (REST API).

Fast-API Core Endpoints, exactly as specified in the class notes:
    /upload-document  - inject data files into the system for the LLM to use
    /ask-questions     - inject a question (prompt) and get a grounded answer
    /health-check      - confirms the API is up, status code = 200

Run with: uvicorn app.api.main:app --host 0.0.0.0 --port 8080
"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import APIError, NotFoundError, RateLimitError
from pydantic import BaseModel

from app.agents.graph import run_agentic_rag
from app.core.config import settings
from app.services.chunking import chunk_documents
from app.services.ingestion import load_document
from app.services.vectorstore import (
    collection_count,
    replace_source_chunks,
    reset_collection,
)
from app.utils.guardrails import log_event, validate_question, validate_upload

app = FastAPI(
    title="GenAI Doc Assistant",
    description="Agentic RAG API for querying enterprise documents.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    trace: list


@app.get("/health-check")
def health_check():
    """API is up and running. Status code = 200."""
    return {"status": "ok", "chunks_indexed": collection_count()}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Injects a data file into the system: saves it, extracts text, chunks
    it, embeds it, and stores it in the vector DB for later retrieval.
    """
    contents = await file.read()
    is_valid, error = validate_upload(file.filename, len(contents))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    save_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        docs = load_document(save_path)
        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError(
                "No readable text was found in the document. Scanned PDFs may require OCR."
            )
        added = replace_source_chunks(os.path.basename(file.filename), chunks)
    except Exception as e:
        log_event("ingestion_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}")

    log_event("upload_document", filename=file.filename, chunks_added=added)
    return {
        "filename": file.filename,
        "chunks_added": added,
        "total_chunks_in_store": collection_count(),
    }


@app.post("/ask-questions", response_model=AskResponse)
def ask_questions(request: AskRequest):
    """
    Injects the user's question (prompt) into the agentic RAG pipeline and
    returns a grounded answer, its sources, and the agent trace.
    """
    is_valid, error = validate_question(request.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    if collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Call /upload-document first.",
        )

    try:
        result = run_agentic_rag(request.question)
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=(
                "OpenAI rate limit or quota reached. Check your API project "
                "billing and limits, then try again."
            ),
            headers={"Retry-After": "60"},
        ) from e
    except NotFoundError as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Configured OpenAI model '{settings.OPENAI_MODEL}' is not "
                "available to this API key. Update OPENAI_MODEL in .env."
            ),
        ) from e
    except APIError as e:
        status_code = getattr(e, "status_code", None)
        log_event(
            "openai_api_error",
            error_type=type(e).__name__,
            status_code=status_code,
            error=str(e),
        )
        provider_status = f" ({status_code})" if status_code else ""
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API request failed{provider_status}: {str(e)}",
        ) from e
    log_event("ask_questions", question=request.question)

    return AskResponse(
        answer=result.get("answer", "No answer generated."),
        sources=result.get("retrieved_chunks", []),
        trace=result.get("trace", []),
    )


@app.post("/reset-index")
def reset_index():
    """Utility endpoint: wipes the vector DB. Handy while testing."""
    reset_collection()
    return {"status": "reset"}
