from pathlib import Path
from typing import Any, Optional

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

def load_pages(file_path: Path, metadata: Optional[dict] = None, **options) -> list:
    """Load a document into ordered `LoadedPage` units via the loader registry.

    This is the page-aware entry point used by the ingestion worker: it keeps
    page and sheet boundaries so chunks can be cited accurately.

    Text cleaning runs here, after the loaders have established page/sheet
    boundaries and before chunking, so normalisation cannot merge pages.
    """
    from src.config.config import ingestion_options
    from src.ingestion.cleaning import clean_pages
    from src.ingestion.loaders import load_document

    merged = {**ingestion_options(), **options}

    # Split options: loaders only understand extraction settings, cleaning only
    # understands its own. Passing one set to the other raises TypeError.
    cleaning_enabled = merged.pop("clean", True)
    cleaning_options = {
        key: merged.pop(key)
        for key in ("remove_furniture", "fix_hyphenation", "strip_page_numbers")
        if key in merged
    }

    pages = load_document(file_path, metadata, **merged)

    if cleaning_enabled:
        pages = clean_pages(pages, **cleaning_options)

    return pages


def extract_text(file_path: Path) -> str:
    """Flatten a document to a single string.

    Retained for backward compatibility. Prefer `load_pages`, which preserves
    page and sheet metadata for citations.
    """
    pages = load_pages(file_path)
    return "\n\n".join(page.text for page in pages if page.text)
