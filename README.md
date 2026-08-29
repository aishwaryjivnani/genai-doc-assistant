# GenAI Doc Assistant — Agentic RAG Capstone Project

A Generative AI application that lets you upload enterprise documents
(PDF/TXT/CSV/Excel/JSON/YAML) and ask questions about them, answered through
an agentic pipeline (Planner → Retriever → Reasoning → Response/Validator)
grounded on a vector database.

Personal learning project — see **Limitations** below.

## Architecture

```
        ┌───────────────┐        ┌──────────────────┐
        │  Streamlit UI │──HTTP─▶│   FastAPI (Task 2)│
        │  (optional)   │        │  /upload-document  │
        └───────────────┘        │  /ask-questions     │
                                  │  /health-check      │
                                  └──────────┬──────────┘
                                             │
                    ┌────────────────────────┼───────────────────────┐
                    ▼                                                ▼
        ┌─────────────────────┐                         ┌──────────────────────┐
        │ Ingestion + Chunking │                         │ LangGraph Agent Flow │
        │ (Task 3 & 4)         │                         │ (Task 8)             │
        └──────────┬───────────┘                         │                      │
                   ▼                                     │  Planner             │
        ┌─────────────────────┐   similarity_search       │     │               │
        │  Chroma Vector DB    │◀──────────────────────── │     ▼               │
        │  (Task 5 & 6)         │──────────────────────▶ │  Retriever ─▶ Chroma │
        └─────────────────────┘                          │     │               │
                                                          │     ▼               │
                                                          │  Reasoning (Gemini) │
                                                          │     │               │
                                                          │     ▼               │
                                                          │  Response/Validator │
                                                          │  (loops back on     │
                                                          │   failure)          │
                                                          └──────────────────────┘
```

## Folder structure

```
genai-doc-assistant/
├── app/
│   ├── api/
│   │   └── main.py          # FastAPI endpoints (Task 2)
│   ├── core/
│   │   └── config.py        # env-based settings (Task 1)
│   ├── services/
│   │   ├── ingestion.py     # Task 3
│   │   ├── chunking.py      # Task 4
│   │   └── vectorstore.py   # Task 5 & 6
│   ├── agents/
│   │   ├── state.py
│   │   └── graph.py         # Task 8 (LangGraph)
│   └── utils/
│       └── guardrails.py    # Task 9
├── data/
│   ├── uploads/              # <Input_files>
│   └── chroma_db/            # persisted vector DB
├── streamlit_app.py           # optional Web UI (Task 2)
├── requirements.txt
├── .env.example
├── .gitignore                 # excludes secrets
├── Dockerfile
└── README.md
```

## Agent roles (Task 8)

| Agent | Responsibility |
|---|---|
| **Planner** | Decides the steps — rewrites the raw question into a focused search query. |
| **Retriever** | Fetches content from the knowledge base (Chroma) via similarity search. |
| **Reasoning** | Analyses the retrieved content and generates a grounded answer (the RAG step). |
| **Response / Validator** | Fact-checks the answer against the retrieved context (the "Result Verification agent" from Task 9); on failure, loops back to Reasoning up to `MAX_REASONING_RETRIES` times. |

## Tech stack

- **API**: FastAPI + Uvicorn
- **UI (optional)**: Streamlit, calling the API over HTTP
- **Vector DB**: ChromaDB (local, persisted to `data/chroma_db/`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **LLM**: Google Gemini free tier (`gemini-2.5-flash` by default) via `langchain-google-genai` — no OpenAI, no Ollama
- **Agent framework**: LangGraph (state graph with a conditional retry loop)
- **Document parsing**: `pypdf` (PDF), `pandas`/`openpyxl` (CSV/Excel), `json`/`pyyaml` (JSON/YAML), plain read (TXT)

## Setup

```bash
# 1. Create the virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
```

Get a **free** Gemini API key at https://aistudio.google.com/apikey, paste it
into `.env` as `GOOGLE_API_KEY`. Google's free-tier model line-up shifts
every few months — check https://ai.google.dev/gemini-api/docs/pricing and
update `GEMINI_MODEL` in `.env` if the model you're using gets retired.

```bash
# 4. Run the API (standalone app)
uvicorn app.api.main:app --host 0.0.0.0 --port 8080
```

Test it:
```bash
curl http://localhost:8080/health-check
curl -F "file=@/path/to/some.pdf" http://localhost:8080/upload-document
curl -X POST http://localhost:8080/ask-questions \
     -H "Content-Type: application/json" \
     -d '{"question": "What does this document say about X?"}'
```

Or open the interactive Swagger docs at `http://localhost:8080/docs`.

**Optional Web UI:**
```bash
# in a second terminal, with the API already running
streamlit run streamlit_app.py
```

## Deployment (Task 10)

Matches the options discussed in class:

1. **Standalone (local)**: `uvicorn app.api.main:app --port 8080` →
   `http://localhost:8080/docs`
2. **Docker**:
   ```bash
   docker build -t genai-doc-assistant .
   docker run -p 8080:8080 --env-file .env genai-doc-assistant
   ```
3. **Cloud (Render — free open-source-friendly tier)**: push this repo to
   GitHub, create a new "Web Service" on render.com pointing at the repo,
   set the start command to `uvicorn app.api.main:app --host 0.0.0.0 --port
   $PORT`, and add `GOOGLE_API_KEY` as an environment variable in the
   dashboard.
4. **Cloud (AWS/Azure/GCP)**: any container hosting service (ECS, App
   Service, Cloud Run) can run the same Docker image.

## Limitations & challenges faced

- **Relevance threshold is heuristic.** `guardrails.has_sufficient_context`
  uses a simple distance cutoff on Chroma's default metric — tune
  `MIN_RELEVANCE_SCORE` against your own documents.
- **No OCR.** Scanned/image-only PDFs won't extract text via `pypdf`.
- **No chat history.** Each question is handled independently.
- **No auth.** Fine for local/personal use, not for a shared deployment.
- **Gemini free-tier rate limits.** The free tier has request-per-minute
  and request-per-day caps; heavy testing (each question can trigger 2–4
  Gemini calls: planner, reasoning, validator, and possibly a retry) can
  hit them. Space out requests during testing.
- **Model name churn.** Google retires/renames free-tier Gemini models every
  few months (e.g. `gemini-2.0-flash` was retired June 2026) — this is the
  main practical challenge of building on a free LLM tier long-term.
  `GEMINI_MODEL` is kept in `.env` specifically so this is a one-line fix.
- **Validator is one LLM checking another.** Reduces obvious hallucinations
  but isn't a formal correctness guarantee.
- **Single shared vector collection.** No per-user/session isolation.

## Possible next steps

- Add conversation memory via LangGraph checkpointing.
- Add a re-ranking step between Retriever and Reasoning.
- Add streaming responses.
- Add automated tests for ingestion edge cases (empty files, huge
  spreadsheets, corrupted PDFs, malformed JSON/YAML).

## Submission

Per the submission guidelines: this folder should be zipped as-is (code +
this README + any sample output you capture) into a single `.zip` and
submitted once.
