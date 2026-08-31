"""
Optional Web UI (Task 2's other option — the class notes list both a REST
API and a Web UI). This is a thin client: every action just calls the
FastAPI endpoints over HTTP, so the UI and the API stay decoupled.

Run the API first: uvicorn app.api.main:app --port 8080
Then run this:      streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")
MAX_UPLOAD_MB = 20

st.set_page_config(page_title="GenAI Doc Assistant", layout="wide")

# ---------------------------------------------------------------------------
# Styling: drag-and-drop upload card look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        background-color: #fafbfc;
        border: 2px dashed #c9cdd3;
        border-radius: 14px;
        padding: 32px 16px;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: #f5a623; }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #f5a623 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }
    [data-testid="stFileUploaderDropzone"] button:hover { background-color: #e0951a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 GenAI Doc Assistant")
st.caption("Personal capstone project — Web UI calling the FastAPI backend.")

# Health check
try:
    health = requests.get(f"{API_BASE}/health-check", timeout=5).json()
    chunks_indexed = health.get("chunks_indexed")
    if chunks_indexed is None:
        st.sidebar.success("API is up")
    else:
        st.sidebar.success(f"API is up — {chunks_indexed} chunks indexed")
except Exception:
    st.sidebar.error(f"Can't reach the API at {API_BASE}. Is uvicorn running?")

with st.sidebar:
    st.divider()
    if st.button("Reset vector DB"):
        requests.post(f"{API_BASE}/reset-index", timeout=30)
        st.rerun()

# ---------------------------------------------------------------------------
# 1. Upload card
# ---------------------------------------------------------------------------
st.header("1. Upload a document")

tab_file, tab_link = st.tabs(["📁 Upload from device", "🔗 Paste a Drive / Dropbox link"])

resp = None

with tab_file:
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "txt", "csv", "xlsx", "xls", "json", "yaml", "yml"],
        label_visibility="collapsed",
    )
    st.caption(f"Max {MAX_UPLOAD_MB} MB · PDF, TXT, CSV, Excel, JSON, YAML")

    if uploaded_file:
        if st.button("Upload file", type="primary", key="btn_file"):
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_BASE}/upload-document", files=files, timeout=120)

with tab_link:
    drive_link = st.text_input(
        "Direct-download URL",
        placeholder="https://drive.google.com/uc?export=download&id=…",
    )
    st.caption("Google Drive: use a direct-download link (File → Share → Copy link, then convert with uc?export=download).")

    if drive_link:
        if st.button("Upload from link", type="primary", key="btn_link"):
            with st.spinner("Fetching and indexing…"):
                try:
                    dl_resp = requests.get(drive_link, timeout=60)
                    dl_resp.raise_for_status()
                    filename = drive_link.split("/")[-1].split("?")[0] or "linked_file"
                    files = {"file": (filename, dl_resp.content)}
                    resp = requests.post(f"{API_BASE}/upload-document", files=files, timeout=120)
                except Exception as e:
                    st.error(
                        f"Couldn't fetch that URL ({e}). Make sure it's a direct-download link."
                    )

if resp is not None:
    if resp.status_code == 200:
        data = resp.json()
        st.success(
            f"✅ **{data['filename']}** indexed — "
            f"{data['chunks_added']} chunks added "
            f"({data['total_chunks_in_store']} total in store)"
        )
    else:
        try:
            st.error(resp.json().get("detail", resp.text))
        except Exception:
            st.error(resp.text or f"Error {resp.status_code}")

st.divider()

# ---------------------------------------------------------------------------
# 2. Ask a question
# ---------------------------------------------------------------------------
st.header("2. Ask a question")
question = st.text_input("Your question about the uploaded documents")

if st.button("Ask", type="primary") and question:
    with st.spinner("Agents working: planner → retriever → reasoning → response..."):
        resp = requests.post(f"{API_BASE}/ask-questions", json={"question": question}, timeout=120)

    if resp.status_code == 200:
        data = resp.json()
        st.subheader("Answer")
        st.write(data["answer"])

        with st.expander("Agent trace"):
            for line in data["trace"]:
                st.write("- " + line)

        with st.expander("Retrieved source chunks"):
            for c in data["sources"]:
                st.markdown(f"**{c['source']}** (distance: {c['score']:.3f})")
                st.text(c["text"][:500])
    else:
        try:
            st.error(resp.json().get("detail", resp.text))
        except Exception:
            st.error(resp.text or f"Error {resp.status_code}")
