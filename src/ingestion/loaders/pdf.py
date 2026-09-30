# src/ingestion/loaders/pdf.py
"""Loader for PDF documents, including scanned PDFs.

Strategy per page:
1. Extract the native text layer (PyMuPDF, falling back to pypdf).
2. If a page yields too little text it is treated as scanned: rasterize it
   and run OCR, then keep whichever extraction produced more text.
3. Optionally extract tables and append them as Markdown.

This "detect then OCR" fallback means mixed documents (a few scanned pages in
an otherwise digital PDF) are handled page by page.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.loaders.capabilities import has_pymupdf, ocr_available
from src.ingestion.loaders.exceptions import LoaderError, OcrNotAvailableError
from src.ingestion.loaders.interface import LoadedPage, base_metadata
from src.ingestion.loaders.markdown import grid_to_markdown
from src.ingestion.loaders.ocr import DEFAULT_DPI, ocr_pdf_page

logger = logging.getLogger(__name__)

# A page with fewer non-whitespace characters than this is assumed scanned.
DEFAULT_MIN_TEXT_CHARS = 40


def _native_text_pymupdf(path: Path) -> list[str] | None:
    """Extract per-page text with PyMuPDF, or None if PyMuPDF is unavailable."""
    if not has_pymupdf():
        return None

    try:
        import pymupdf
    except ImportError:  # pragma: no cover - legacy module name
        import fitz as pymupdf

    pages: list[str] = []
    try:
        with pymupdf.open(str(path)) as document:
            for page in document:
                pages.append(page.get_text("text") or "")
    except Exception as exc:
        raise LoaderError(f"Could not read PDF {path.name}: {exc}") from exc
    return pages


def _native_text_pypdf(path: Path) -> list[str]:
    """Extract per-page text with pypdf, tolerating blank or broken pages."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise LoaderError(
            "PDF ingestion requires either 'pymupdf' or 'pypdf'"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise LoaderError(f"Could not open PDF {path.name}: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("Failed to extract a page from %s: %s", path.name, exc)
            pages.append("")
    return pages


def extract_page_tables(page: Any) -> list[str]:
    """Extract tables from a PyMuPDF page as Markdown strings.

    Uses PyMuPDF's built-in table finder (`find_tables`), available in
    PyMuPDF >= 1.23, so no extra dependency is needed.
    """
    try:
        finder = page.find_tables()
    except Exception as exc:
        logger.debug("Table detection unavailable on this page: %s", exc)
        return []

    tables: list[str] = []
    try:
        found = list(getattr(finder, "tables", []) or [])
    except Exception as exc:  # pragma: no cover - malformed table structure
        logger.debug("Could not iterate tables: %s", exc)
        return []

    for table in found:
        try:
            grid = table.extract()
        except Exception as exc:
            logger.debug("Could not extract table data: %s", exc)
            continue
        markdown = grid_to_markdown(grid or [])
        if markdown:
            tables.append(markdown)
    return tables


def load_pdf(
    path: Path,
    metadata: dict[str, Any] | None = None,
    *,
    ocr: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = DEFAULT_DPI,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    extract_tables: bool = True,
) -> list[LoadedPage]:
    """Load a PDF into one `LoadedPage` per page.

    Args:
        path: PDF file to read.
        metadata: Caller metadata merged into every page's metadata.
        ocr: Whether to OCR pages that have no usable text layer.
        ocr_language: Tesseract language code.
        ocr_dpi: Rasterization resolution for OCR.
        min_text_chars: Non-whitespace character count below which a page is
            considered scanned and OCR is attempted.
        extract_tables: Whether to pull tables out as Markdown.

    Raises:
        LoaderError: The PDF could not be read at all.
        ScannedDocumentError: The PDF is scanned and OCR is unavailable.
    """
    # Normalise so the loader behaves the same whether called via
    # `load_document` or directly with a string path.
    path = Path(path)

    native_pages = _native_text_pymupdf(path)
    if native_pages is None:
        native_pages = _native_text_pypdf(path)
    page_count = len(native_pages)
    if page_count == 0:
        logger.warning("PDF %s contains no pages", path.name)
        return []

    pages: list[LoadedPage] = []
    ocr_pages = 0
    scanned_pages: list[int] = []

    # Reuse a single PyMuPDF handle for both rasterization and tables.
    document = None
    if has_pymupdf() and (ocr or extract_tables):
        try:
            import pymupdf

            document = pymupdf.open(str(path))
        except ImportError:  # pragma: no cover - legacy module name
            import fitz as pymupdf

            document = pymupdf.open(str(path))
        except Exception as exc:
            logger.warning("Could not reopen %s for OCR/tables: %s", path.name, exc)
            document = None

    try:
        for number, native_text in enumerate(native_pages, start=1):
            text = native_text or ""
            extraction = "native"
            significant = len("".join(text.split()))

            # Tables are tried first: a table can supply text for a page whose
            # prose layer is thin, which would otherwise trigger needless OCR.
            if extract_tables and document is not None:
                table_markdown = extract_page_tables(document[number - 1])
                if table_markdown:
                    text = text.rstrip() + "\n\n" + "\n\n".join(table_markdown)
                    significant = len("".join(text.split()))

            # Detect a scan independently of whether OCR is switched on, so a
            # text-less page is always reported rather than silently skipped.
            # NOTE: keep the parentheses -- `a and b < c` parses as
            # `(a and b) < c`, which is not the intended test.
            looks_scanned = significant < min_text_chars
            if looks_scanned:
                scanned_pages.append(number)

                if not ocr:
                    logger.info(
                        "Page %d of %s has little text and OCR is disabled",
                        number,
                        path.name,
                    )
                elif ocr_available() and document is not None:
                    try:
                        ocr_text = ocr_pdf_page(
                            document[number - 1],
                            language=ocr_language,
                            dpi=ocr_dpi,
                        )
                    except OcrNotAvailableError:
                        raise
                    except LoaderError as exc:
                        logger.warning(
                            "OCR failed on %s page %d: %s", path.name, number, exc
                        )
                        ocr_text = ""

                    # Keep whichever route produced more usable text.
                    if len("".join(ocr_text.split())) > significant:
                        text = ocr_text
                        extraction = "ocr"
                        ocr_pages += 1
                else:
                    logger.info(
                        "Page %d of %s looks scanned but OCR is unavailable",
                        number,
                        path.name,
                    )

            page_metadata = {
                **base_metadata(path, metadata),
                "page": number,
                "page_count": page_count,
            }

            pages.append(
                LoadedPage(
                    text=text.strip(),
                    index=number - 1,
                    metadata=page_metadata,
                    extraction=extraction,
                )
            )
    finally:
        if document is not None:
            document.close()

    # A PDF where no page produced text could not be read usefully. Say so
    # rather than indexing a set of empty chunks.
    if not any(not p.is_empty for p in pages) and scanned_pages:
        from src.ingestion.loaders.exceptions import ScannedDocumentError

        if not ocr:
            raise ScannedDocumentError(
                f"{path.name} has no extractable text layer and appears to be "
                f"scanned. OCR is disabled, so this file cannot be ingested. "
                f"Enable ingestion.ocr.enabled in the configuration to read it."
            )
        if not ocr_available():
            raise ScannedDocumentError(
                f"{path.name} appears to be a scanned PDF with no text layer, "
                f"and OCR is unavailable. See the worker log for install "
                f"instructions."
            )

    if ocr_pages:
        logger.info("OCR extracted text from %d page(s) of %s", ocr_pages, path.name)

    # Warn only about pages that are genuinely text-less. A page slightly below
    # the threshold is a short page, not a scan, so do not flag it.
    empty_pages = [p.metadata["page"] for p in pages if p.is_empty]
    if empty_pages:
        logger.warning(
            "%d page(s) of %s produced no text at all (pages %s); they are "
            "likely scans and contributed nothing to the index",
            len(empty_pages),
            path.name,
            ", ".join(str(n) for n in empty_pages),
        )

    return pages
