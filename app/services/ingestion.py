"""
Task 3: Document ingestion.

Supports every format called out in the class notes: TXT, PDF, CSV, Excel,
JSON, YAML. Turns each into LangChain Document objects ready for chunking.
"""
import json
import os
from typing import List

import pandas as pd
import yaml
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: str) -> List[Document]:
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i + 1},
                )
            )
    return docs


def load_txt(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [
        Document(
            page_content=text,
            metadata={"source": os.path.basename(path), "structured": False},
        )
    ]


def load_csv(path: str) -> List[Document]:
    df = pd.read_csv(path)
    return _dataframe_to_documents(df, path)


def load_excel(path: str) -> List[Document]:
    sheets = pd.read_excel(path, sheet_name=None)  # dict of {sheet_name: df}
    docs = []
    for sheet_name, df in sheets.items():
        docs.extend(_dataframe_to_documents(df, path, sheet_name=sheet_name))
    return docs


def load_json(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    source = os.path.basename(path)

    if isinstance(data, list):
        docs = []

        # --- Summary document ---
        # Pre-compute unique values for every scalar field so that queries like
        # "list all unique aisle IDs" are answered from a single retrieved chunk
        # rather than needing to scan every record.
        if data and isinstance(data[0], dict):
            all_keys = {k for record in data for k in record}
            lines = [
                f"Summary of {source}",
                f"Total records: {len(data)}",
            ]
            for key in sorted(all_keys):
                unique_vals = sorted(
                    {record[key] for record in data if key in record and not isinstance(record[key], (dict, list))},
                    key=lambda v: (str(type(v)), v),
                )
                lines.append(f"Unique {key} values ({len(unique_vals)}): {unique_vals}")
            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "source": source,
                        "record_type": "summary",
                        "structured": True,
                    },
                )
            )

        # --- Batched record documents ---
        records_per_chunk = 10
        for start in range(0, len(data), records_per_chunk):
            batch = data[start : start + records_per_chunk]
            docs.append(
                Document(
                    page_content=json.dumps(batch, indent=2),
                    metadata={
                        "source": source,
                        "record_start": start,
                        "record_end": start + len(batch),
                        "structured": True,
                    },
                )
            )
        return docs

    return [
        Document(
            page_content=json.dumps(data, indent=2),
            metadata={"source": source, "structured": True},
        )
    ]


def load_yaml(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        Document(
            page_content=yaml.dump(data, default_flow_style=False),
            metadata={"source": os.path.basename(path), "structured": True},
        )
    ]


def _dataframe_to_documents(df: pd.DataFrame, path: str, sheet_name: str = None) -> List[Document]:
    """
    Turns tabular data into text blocks a few rows at a time, so retrieval
    can find specific rows without losing column context.
    """
    docs = []
    rows_per_chunk = 10
    for start in range(0, len(df), rows_per_chunk):
        chunk_df = df.iloc[start : start + rows_per_chunk]
        text = chunk_df.to_string(index=False)
        meta = {
            "source": os.path.basename(path),
            "row_start": start,
            "row_end": start + len(chunk_df),
            "structured": True,
        }
        if sheet_name:
            meta["sheet"] = sheet_name
        docs.append(Document(page_content=text, metadata=meta))
    return docs


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".csv": load_csv,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


def load_document(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    return LOADERS[ext](path)
