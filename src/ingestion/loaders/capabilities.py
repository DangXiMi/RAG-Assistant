# src/ingestion/loaders/capabilities.py
"""Runtime detection of optional ingestion dependencies.

Every optional capability is probed lazily and cached. Callers use these
helpers to degrade gracefully instead of raising ImportError at module
import time.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
from functools import lru_cache

logger = logging.getLogger(__name__)


def _module_available(name: str) -> bool:
    """Return True if `name` can be imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=None)
def has_pymupdf() -> bool:
    """PyMuPDF: PDF text extraction, table extraction and page rasterization."""
    return _module_available("pymupdf") or _module_available("fitz")


def _import_pymupdf():
    """Import PyMuPDF under either its new or legacy module name."""
    try:
        return importlib.import_module("pymupdf")
    except ImportError:
        return importlib.import_module("fitz")


@lru_cache(maxsize=None)
def has_openpyxl() -> bool:
    """openpyxl: .xlsx / .xlsm reading."""
    return _module_available("openpyxl")


@lru_cache(maxsize=None)
def has_pytesseract() -> bool:
    """The pytesseract Python wrapper for the Tesseract OCR engine."""
    return _module_available("pytesseract")


@lru_cache(maxsize=None)
def has_tesseract_binary() -> bool:
    """The Tesseract executable itself, which pytesseract shells out to.

    Checks the configured path first, then PATH.
    """
    import os

    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        return os.path.isfile(configured) or shutil.which(configured) is not None
    return shutil.which("tesseract") is not None


def ocr_backend() -> str | None:
    """Name of the usable OCR backend, or None when OCR is unavailable."""
    if has_pytesseract() and has_tesseract_binary():
        return "tesseract"
    return None


def ocr_available() -> bool:
    """True when OCR can actually run (wrapper *and* engine binary present)."""
    return ocr_backend() is not None


def ocr_unavailable_reason() -> str:
    """Human-readable explanation of why OCR is unavailable."""
    if not has_pytesseract() and not has_tesseract_binary():
        return (
            "the 'pytesseract' package is not installed and the Tesseract "
            "executable was not found"
        )
    if not has_pytesseract():
        return "the 'pytesseract' package is not installed"
    if not has_tesseract_binary():
        return (
            "the Tesseract executable was not found. Install it "
            "(e.g. 'winget install UB-Mannheim.TesseractOCR' on Windows, "
            "'apt-get install tesseract-ocr' on Debian/Ubuntu) or set the "
            "TESSERACT_CMD environment variable to its full path"
        )
    return "unknown reason"


def summary() -> dict[str, bool]:
    """Snapshot of optional capability availability, for logs and diagnostics."""
    return {
        "pdf_tables (pymupdf)": has_pymupdf(),
        "excel (openpyxl)": has_openpyxl(),
        "ocr (pytesseract)": has_pytesseract(),
        "ocr (tesseract binary)": has_tesseract_binary(),
        "ocr (usable)": ocr_available(),
    }


def log_summary() -> None:
    """Log the capability snapshot once at startup."""
    for name, available in summary().items():
        logger.info(
            "Ingestion capability %-24s : %s",
            name,
            "available" if available else "MISSING",
        )
    if not ocr_available():
        logger.warning("OCR unavailable: %s", ocr_unavailable_reason())
