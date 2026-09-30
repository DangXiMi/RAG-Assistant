# src/ingestion/loaders/__init__.py
"""Multi-format document loaders.

`load_document` is the single entry point: it dispatches on file extension and
returns an ordered list of `LoadedPage` units, each carrying the metadata
needed for citation (`source`, `page` / `sheet`).

Supported formats:
    PDF   .pdf                     native text, tables, scanned-page OCR
    Word  .docx                    paragraphs, headings, tables (Markdown)
    Excel .xlsx .xlsm              one unit per sheet, Markdown tables
    CSV   .csv                     single unit, Markdown table
    HTML  .html .htm               block structure, tables (Markdown)
    Text  .txt .md .markdown       verbatim
    Image .png .jpg .jpeg ...      OCR
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from src.ingestion.loaders import capabilities
from src.ingestion.loaders.exceptions import (
    LoaderError,
    MissingDependencyError,
    OcrNotAvailableError,
    ScannedDocumentError,
    UnsupportedFormatError,
)
from src.ingestion.loaders.interface import (
    LOCATION_KEY,
    PAGE_KEY,
    SHEET_KEY,
    SOURCE_KEY,
    DocumentLoader,
    LoadedPage,
    base_metadata,
)

logger = logging.getLogger(__name__)

# Extension -> loader. Kept in one place so the API and UI can advertise the
# same list the dispatcher actually supports.
LOADER_REGISTRY: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",  # legacy .doc will fail to open; reported clearly
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
    ".gif": "image",
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(LOADER_REGISTRY))

# Extensions that always require OCR, used for pre-flight validation so the
# API can reject an upload immediately instead of failing it in the worker.
OCR_EXTENSIONS: tuple[str, ...] = tuple(
    ext for ext, kind in LOADER_REGISTRY.items() if kind == "image"
)

# Human-readable groups, for UI hints and error messages.
FORMAT_DESCRIPTIONS: dict[str, str] = {
    "PDF": (".pdf",),
    "Word": (".docx",),
    "Excel": (".xlsx", ".xlsm", ".xls"),
    "CSV": (".csv",),
    "HTML": (".html", ".htm"),
    "Text": (".txt", ".md", ".markdown"),
    "Image (OCR)": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"),
}


def supported_extension_list(include_dot: bool = False) -> list[str]:
    """Supported extensions, optionally without the leading dot.

    Upload widgets want `["pdf", "docx", ...]`; validation wants the dotted form.
    """
    if include_dot:
        return list(SUPPORTED_EXTENSIONS)
    return [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]


def is_supported(path: Path | str) -> bool:
    """True when a filename's extension has a registered loader."""
    suffix = Path(path).suffix.lower()
    return suffix in LOADER_REGISTRY


def load_document(
    path: Path | str,
    metadata: dict[str, Any] | None = None,
    *,
    ocr: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    min_text_chars: int = 40,
    extract_tables: bool = True,
) -> list[LoadedPage]:
    """Load any supported document into ordered, citable page units.

    Args:
        path: File to load.
        metadata: Extra metadata merged into every unit (e.g. `document_id`).
        ocr: Enable OCR for images and text-less PDF pages.
        ocr_language: Tesseract language code.
        ocr_dpi: Rasterization resolution for OCR.
        min_text_chars: PDF page text threshold below which OCR is attempted.
        extract_tables: Extract tables as Markdown where the format supports it.

    Returns:
        Ordered `LoadedPage` units. Empty when the file holds no usable text.

    Raises:
        UnsupportedFormatError: No loader is registered for the extension.
        MissingDependencyError: An optional dependency for the format is absent.
        OcrNotAvailableError: OCR was required but is not installed.
        ScannedDocumentError: A scanned PDF was found and OCR is unavailable.
        LoaderError: The file could not be read.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise LoaderError(f"File not found: {path}")

    suffix = path.suffix.lower()
    kind = LOADER_REGISTRY.get(suffix)
    if kind is None:
        raise UnsupportedFormatError(suffix or "(no extension)", list(SUPPORTED_EXTENSIONS))

    loader: Callable[..., list[LoadedPage]]

    if kind == "pdf":
        from src.ingestion.loaders.pdf import load_pdf

        loader = lambda p, m: load_pdf(  # noqa: E731 - local binding
            p,
            m,
            ocr=ocr,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            min_text_chars=min_text_chars,
            extract_tables=extract_tables,
        )
    elif kind == "docx":
        from src.ingestion.loaders.docx import load_docx

        loader = load_docx
    elif kind == "excel":
        from src.ingestion.loaders.excel import load_excel

        loader = load_excel
    elif kind == "csv":
        from src.ingestion.loaders.excel import load_csv

        loader = load_csv
    elif kind == "html":
        from src.ingestion.loaders.text_loaders import load_html

        loader = load_html
    elif kind == "text":
        from src.ingestion.loaders.text_loaders import load_text

        loader = load_text
    elif kind == "image":
        from src.ingestion.loaders.image import load_image

        loader = lambda p, m: load_image(p, m, ocr_language=ocr_language)  # noqa: E731
    else:  # pragma: no cover - registry and dispatch are kept in sync
        raise UnsupportedFormatError(suffix, list(SUPPORTED_EXTENSIONS))

    pages = loader(path, metadata or {})
    logger.info(
        "Loaded %s (%s): %d unit(s), %d chars total, methods=%s",
        path.name,
        kind,
        len(pages),
        sum(p.char_count for p in pages),
        sorted({p.extraction for p in pages}) or ["none"],
    )
    return pages


__all__ = [
    "LOADER_REGISTRY",
    "SUPPORTED_EXTENSIONS",
    "OCR_EXTENSIONS",
    "FORMAT_DESCRIPTIONS",
    "DocumentLoader",
    "LoadedPage",
    "LoaderError",
    "MissingDependencyError",
    "OcrNotAvailableError",
    "ScannedDocumentError",
    "UnsupportedFormatError",
    "SOURCE_KEY",
    "PAGE_KEY",
    "SHEET_KEY",
    "LOCATION_KEY",
    "base_metadata",
    "capabilities",
    "is_supported",
    "load_document",
    "supported_extension_list",
]
