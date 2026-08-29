"""
Task 2: User interaction layer (REST API).

Fast-API Core Endpoints, exactly as specified in the class notes:
    /upload-document  - inject data files into the system for the LLM to use
    /ask-questions     - inject a question (prompt) and get a grounded answer
    /health-check      - confirms the API is up, status code = 200

Run with: uvicorn app.api.main:app --host 0.0.0.0 --port 8080
"""
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.agents.graph import run_agentic_rag
from app.core.config import settings
from app.services.chunking import chunk_documents
from app.services.ingestion import load_document
from app.services.vectorstore import add_chunks, collection_count, reset_collection
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
        added = add_chunks(chunks)
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

    result = run_agentic_rag(request.question)
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
