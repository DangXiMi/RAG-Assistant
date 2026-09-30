# src/generation/citations.py
"""Turn retrieved chunk metadata into human-readable citations.

A retrieved chunk knows exactly where it came from: its `metadata` carries the
`source` filename plus either a `page` (PDF/image) or a `sheet` (spreadsheet),
because the loaders emit page-scoped units. This module renders that into the
`report.pdf, p.3` form the brief asks for.

Used by the generator (to label context and build the source list) and by the
UI, so both agree on how a citation reads.
"""
from __future__ import annotations

from typing import Any, Iterable

# Metadata keys consulted, in priority order, when describing a location.
_SOURCE_KEYS = ("source", "filename", "file_name", "doc_name")


def source_name(metadata: dict[str, Any] | None) -> str:
    """Filename a chunk came from, or `"unknown"` when absent."""
    metadata = metadata or {}
    for key in _SOURCE_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def location(metadata: dict[str, Any] | None) -> str:
    """Location within the document: `p.3`, `sheet 'Q2'`, or `""`.

    Page takes priority over sheet; a row range is appended when known.
    """
    metadata = metadata or {}

    page = metadata.get("page")
    if page not in (None, ""):
        return f"p.{page}"

    sheet = metadata.get("sheet")
    if sheet not in (None, ""):
        return f"sheet '{sheet}'"

    # CSV has neither a page nor a sheet name; it may still carry a row range.
    row = metadata.get("row") or metadata.get("location")
    if row:
        return f"rows {row}"

    return ""


def label(metadata: dict[str, Any] | None, prefix: str | None = None) -> str:
    """Full citation label, e.g. `refund_policy.pdf, p.3`.

    Args:
        metadata: Chunk metadata.
        prefix: Optional `[1]`-style marker placed in front.
    """
    name = source_name(metadata)
    where = location(metadata)
    text = f"{name}, {where}" if where else name
    return f"[{prefix}] {text}" if prefix else text


def build_source_refs(docs: Iterable[Any]) -> list[dict[str, Any]]:
    """Structured source references for one retrieval result.

    Args:
        docs: LangChain `Document` objects, or mappings with `metadata`.

    Returns:
        One entry per document: `{"ref", "source", "location", "page",
        "sheet", "chunk_id", "score", "label"}`. Entries are de-duplicated on
        `(source, location)` while keeping the first (best-ranked) occurrence,
        because the same page can legitimately be retrieved as several chunks
        and repeating it adds no information for a reader.
    """
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for index, doc in enumerate(docs, start=1):
        if isinstance(doc, dict):
            metadata = doc.get("metadata") or {}
            score = doc.get("score")
        else:
            metadata = getattr(doc, "metadata", {}) or {}
            score = metadata.get("score")

        name = source_name(metadata)
        where = location(metadata)
        key = (name, where)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "ref": index,
                "source": name,
                "location": where,
                "page": metadata.get("page"),
                "sheet": metadata.get("sheet"),
                "chunk_id": metadata.get("doc_id") or metadata.get("id"),
                "score": score,
                "label": label(metadata, prefix=str(index)),
            }
        )

    return refs


def source_labels(docs: Iterable[Any]) -> list[str]:
    """Flat list of citation labels, e.g. `["refund_policy.pdf, p.3"]`.

    Kept as plain strings so the existing `QueryResponse.sources` contract
    (`list[str]`) is preserved.
    """
    return [ref["label"] for ref in build_source_refs(docs)]
