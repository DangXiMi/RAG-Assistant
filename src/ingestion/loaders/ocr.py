# src/ingestion/loaders/ocr.py
"""OCR support for images and scanned PDF pages.

The backend is Tesseract via `pytesseract`. Two capabilities are required and
both are checked: the wrapper package and the Tesseract executable itself.
When either is missing the callers raise `OcrNotAvailableError` with install
instructions, and the ingest job fails with an actionable message rather than
an obscure traceback.

PDF rasterization uses PyMuPDF, which avoids the extra Poppler dependency
that `pdf2image` requires.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.ingestion.loaders.capabilities import (
    has_pymupdf,
    has_pytesseract,
    has_tesseract_binary,
    ocr_available,
    ocr_backend,
    ocr_unavailable_reason,
)
from src.ingestion.loaders.exceptions import LoaderError, OcrNotAvailableError

logger = logging.getLogger(__name__)

# Rendering resolution for rasterizing PDF pages before OCR. 300 DPI is the
# usual sweet spot for Tesseract accuracy without excessive memory use.
DEFAULT_DPI = 300

# Tesseract page segmentation mode 3 = fully automatic, no OSD.
DEFAULT_PSM = 3

_TESSERACT_INSTALL_HINT = (
    "Install the Tesseract engine (Windows: 'winget install "
    "UB-Mannheim.TesseractOCR'; Debian/Ubuntu: 'apt-get install "
    "tesseract-ocr'; macOS: 'brew install tesseract'), then restart the "
    "worker. If it is installed in a non-standard location, set TESSERACT_CMD "
    "to the full path of the executable."
)


def ensure_ocr_ready() -> None:
    """Raise `OcrNotAvailableError` if OCR cannot run right now."""
    if ocr_available():
        return

    if not has_pytesseract():
        raise OcrNotAvailableError(
            "OCR",
            "pytesseract",
            "Install it with: pip install pytesseract. "
            + _TESSERACT_INSTALL_HINT,
        )

    if not has_tesseract_binary():
        raise OcrNotAvailableError("OCR", "tesseract", _TESSERACT_INSTALL_HINT)

    raise OcrNotAvailableError("OCR", "tesseract", ocr_unavailable_reason())


def _configure_tesseract() -> Any:
    """Import pytesseract and point it at an explicitly configured binary."""
    import pytesseract

    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    return pytesseract


def ocr_image(
    image: Any,
    language: str = "eng",
    psm: int = DEFAULT_PSM,
) -> str:
    """Run OCR over a PIL image and return the recognised text.

    Args:
        image: A PIL `Image` instance.
        language: Tesseract language code, e.g. `"eng"`, `"vie"`, `"eng+vie"`.
        psm: Tesseract page segmentation mode.
    """
    ensure_ocr_ready()
    pytesseract = _configure_tesseract()

    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(image, lang=language, config=config)
    except Exception as exc:
        # Tesseract reports missing language packs and similar as generic
        # errors; surface them as loader errors with context.
        raise LoaderError(f"OCR failed: {exc}") from exc

    return text or ""


def ocr_image_file(
    path: Path,
    language: str = "eng",
    psm: int = DEFAULT_PSM,
) -> str:
    """Run OCR directly over an image file on disk."""
    ensure_ocr_ready()

    try:
        from PIL import Image
    except ImportError as exc:
        raise OcrNotAvailableError(
            "Image OCR", "Pillow", "Install it with: pip install Pillow"
        ) from exc

    try:
        with Image.open(path) as image:
            # Convert to RGB: Tesseract cannot read palette/alpha modes directly.
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            return ocr_image(image, language=language, psm=psm)
    except OcrNotAvailableError:
        raise
    except LoaderError:
        raise
    except Exception as exc:
        raise LoaderError(f"Could not OCR image {path.name}: {exc}") from exc


def pages_to_images(page: Any, dpi: int = DEFAULT_DPI) -> Any:
    """Rasterize one PyMuPDF page into a PIL image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow ships with PyMuPDF use
        raise OcrNotAvailableError(
            "PDF OCR", "Pillow", "Install it with: pip install Pillow"
        ) from exc

    import io

    # PyMuPDF renders at 72 DPI by default; scale to reach the target DPI.
    zoom = dpi / 72.0
    matrix = None
    try:
        import pymupdf

        matrix = pymupdf.Matrix(zoom, zoom)
    except ImportError:  # pragma: no cover - legacy module name
        import fitz

        matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png")))


def ocr_pdf_page(
    page: Any,
    language: str = "eng",
    dpi: int = DEFAULT_DPI,
) -> str:
    """Rasterize one PDF page and OCR it.

    `page` is a PyMuPDF page object (kept untyped so PyMuPDF stays optional).
    """
    if not has_pymupdf():
        raise OcrNotAvailableError(
            "Scanned PDF OCR",
            "pymupdf",
            "Install it with: pip install pymupdf",
        )

    ensure_ocr_ready()
    image = pages_to_images(page, dpi=dpi)
    try:
        return ocr_image(image, language=language)
    finally:
        image.close()


def describe_backend() -> str:
    """Short description of the active OCR backend, for logs and status."""
    backend = ocr_backend()
    if backend is None:
        return f"unavailable ({ocr_unavailable_reason()})"
    return backend
