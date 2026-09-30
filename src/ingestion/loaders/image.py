# src/ingestion/loaders/image.py
"""Loader for standalone image files, via OCR."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.loaders.interface import LoadedPage, base_metadata
from src.ingestion.loaders.ocr import DEFAULT_DPI, ocr_image_file

logger = logging.getLogger(__name__)

# Formats Pillow can open and Tesseract can read well.
SUPPORTED_IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
)


def load_image(
    path: Path,
    metadata: dict[str, Any] | None = None,
    *,
    ocr_language: str = "eng",
) -> list[LoadedPage]:
    """OCR an image file into a single citable unit.

    Raises:
        OcrNotAvailableError: No usable OCR backend is installed.
        LoaderError: The image could not be read or OCR failed.
    """
    text = ocr_image_file(path, language=ocr_language)

    if not text.strip():
        logger.warning("OCR found no text in image %s", path.name)

    page_metadata = base_metadata(path, metadata)
    # An image has no pagination; mark it so citations read "<file>, p.1".
    page_metadata["page"] = 1
    page_metadata["page_count"] = 1
    page_metadata["ocr"] = True

    return [
        LoadedPage(
            text=text.strip(),
            index=0,
            metadata=page_metadata,
            extraction="ocr",
        )
    ]


__all__ = ["load_image", "SUPPORTED_IMAGE_EXTENSIONS", "DEFAULT_DPI"]
