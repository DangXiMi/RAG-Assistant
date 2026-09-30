# src/ingestion/loaders/interface.py
"""The contract every document loader implements.

A loader converts one file into an ordered list of `LoadedPage` units. Each
unit is one *citable location* in the source document: a PDF page, a
spreadsheet sheet, or the document body for flat formats. Downstream code
chunks each unit separately, so the unit's metadata (`source`, `page`,
`sheet`) lands on every chunk and can be cited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Metadata keys that identify where a chunk came from. Used by the citation
# formatter in the generation layer.
SOURCE_KEY = "source"
PAGE_KEY = "page"
SHEET_KEY = "sheet"
LOCATION_KEY = "location"


@dataclass
class LoadedPage:
    """One citable unit of an ingested document.

    Attributes:
        text: Extracted text, with tables already rendered as Markdown.
        index: 0-based position of this unit within the document.
        metadata: Location metadata (`source`, and `page` or `sheet`).
        char_count: Length of `text`, cached for filtering decisions.
        extraction: How the text was obtained -- `"text"`, `"native"`,
            `"ocr"` or `"mixed"`. Useful for debugging scan detection.
    """

    text: str
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    extraction: str = "text"

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text or "")

    @property
    def is_empty(self) -> bool:
        """True when no usable text was extracted from this unit."""
        return not (self.text or "").strip()

    @property
    def label(self) -> str:
        """Human-readable location, e.g. `report.pdf, p.3` or `book.xlsx, sheet 'Q2'`."""
        source = self.metadata.get(SOURCE_KEY, "unknown")
        if PAGE_KEY in self.metadata:
            return f"{source}, p.{self.metadata[PAGE_KEY]}"
        if SHEET_KEY in self.metadata:
            return f"{source}, sheet '{self.metadata[SHEET_KEY]}'"
        return str(source)

    def location(self) -> str:
        """Short location string used in citations."""
        if PAGE_KEY in self.metadata:
            return f"p.{self.metadata[PAGE_KEY]}"
        if SHEET_KEY in self.metadata:
            return f"sheet '{self.metadata[SHEET_KEY]}'"
        return ""


@runtime_checkable
class DocumentLoader(Protocol):
    """A callable that turns a file into ordered `LoadedPage` units.

    Implementations raise `LoaderError` subclasses on failure rather than
    leaking library-specific exceptions.
    """

    def __call__(self, path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
        ...


def base_metadata(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the metadata every loader starts from.

    `source` is the citation-critical filename. Uploads are stored on disk as
    `<job_id>_<original name>`, so the caller can pass `original_filename` to
    keep citations readable; otherwise the on-disk name is used.

    `extra` (the upload metadata from the API layer) is merged on top so
    callers can attach `document_id` or `content_type`, but `source` is always
    derived from the filename and never from caller-supplied data.
    """
    meta: dict[str, Any] = {}

    original = None
    if extra:
        original = extra.get("original_filename")
        meta.update(extra)

    # Prefer the true upload name; fall back to the file on disk.
    meta[SOURCE_KEY] = str(original) if original else path.name
    return meta
