from pathlib import Path
from typing import Any
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

def RRF(top_k, rrf_k, retrievers):
    fused_docs = {}

    for retriever in retrievers :
        for rank, doc in enumerate(retriever):
            doc_id = str(doc["id"])

            if doc_id not in fused_docs:
                fused_docs[doc_id] = {
                    "id": doc_id,
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": 0.0,
                }

            fused_docs[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

    results = sorted(
        fused_docs.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]

def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".html" or ext == ".htm":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    else:
        # Assume plain text
        return file_path.read_text(encoding="utf-8")
