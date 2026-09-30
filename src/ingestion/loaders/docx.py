# src/ingestion/loaders/docx.py
"""Loader for Word .docx documents.

Walks the document body in XML order so paragraphs and tables appear in the
text exactly where they appear in the file, and renders tables as Markdown.
(The previous implementation read only `doc.paragraphs`, which silently
discarded every table in the document.)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.loaders.exceptions import LoaderError
from src.ingestion.loaders.interface import LoadedPage, base_metadata
from src.ingestion.loaders.markdown import grid_to_markdown

logger = logging.getLogger(__name__)

# Word style names that map to Markdown heading levels.
_HEADING_PREFIX = "Heading "


def _paragraph_markdown(paragraph: Any) -> str:
    """Render a paragraph as Markdown, preserving heading levels and lists."""
    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = ""
    try:
        style_name = paragraph.style.name or ""
    except (AttributeError, KeyError):
        style_name = ""

    if style_name.startswith(_HEADING_PREFIX):
        level = style_name[len(_HEADING_PREFIX):].strip()
        try:
            depth = max(1, min(6, int(level)))
        except ValueError:
            depth = 1
        return f"{'#' * depth} {text}"

    if style_name in ("Title",):
        return f"# {text}"
    if style_name in ("Subtitle",):
        return f"## {text}"

    if style_name.startswith("List Bullet"):
        return f"- {text}"
    if style_name.startswith("List Number"):
        return f"1. {text}"

    return text


def _table_markdown(table: Any) -> str:
    """Render a Word table as a Markdown table.

    Reads `table.rows` rather than the inner XML so merged cells are handled
    by python-docx; nested tables are flattened to their text content.
    """
    grid: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            # A merged cell repeats; take the text once per grid position.
            cell_text = " ".join(cell.text.split())
            cells.append(cell_text)
        if any(cells):
            grid.append(cells)

    return grid_to_markdown(grid)


def _part_text(part: Any) -> str:
    """Flatten header/footer text to a single line.

    `part.paragraphs` can yield nested lists when the header contains a table,
    so each entry's `.text` is not guaranteed to be a string. Guarding on the
    type keeps one odd header from silently dropping the whole part.
    """
    chunks: list[str] = []
    try:
        paragraphs = part.paragraphs
    except Exception:  # pragma: no cover - malformed part
        return ""

    for paragraph in paragraphs:
        text = getattr(paragraph, "text", "")
        if isinstance(text, str) and text.strip():
            chunks.append(" ".join(text.split()))

    # Headers sometimes hold only a table, which paragraphs does not expose.
    if not chunks:
        try:
            for table in part.tables:
                markdown = _table_markdown(table)
                if markdown:
                    chunks.append(markdown)
        except Exception:  # pragma: no cover - malformed part
            pass

    return " ".join(chunks).strip()


def load_docx(path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
    """Load a .docx file as a single citable unit, tables included."""
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover - python-docx is a hard dep
        raise LoaderError(
            "Word ingestion requires the 'python-docx' package"
        ) from exc

    try:
        document = Document(str(path))
    except Exception as exc:
        raise LoaderError(f"Could not open Word document {path.name}: {exc}") from exc

    # Iterate the body in document order. python-docx's `paragraphs` and
    # `tables` collections each skip the other's elements, so walking the XML
    # body is the only way to keep true ordering.
    body = document.element.body
    blocks: list[str] = []
    table_count = 0

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            text = _paragraph_markdown(Paragraph(child, document))
            if text:
                blocks.append(text)
        elif tag == "tbl":
            markdown = _table_markdown(Table(child, document))
            if markdown:
                table_count += 1
                blocks.append(f"\n{markdown}\n")

    # Headers and footers are separate parts; keep them, they often carry
    # document titles and version numbers.
    for section in document.sections:
        for part, label in (
            (section.header, "Header"),
            (section.footer, "Footer"),
        ):
            text = _part_text(part)
            if text:
                blocks.append(f"[{label}] {text}")

    text = "\n\n".join(blocks)
    if not text.strip():
        logger.warning("No text extracted from Word document %s", path.name)

    page_metadata = base_metadata(path, metadata)
    page_metadata["tables"] = table_count

    return [
        LoadedPage(
            text=text,
            index=0,
            metadata=page_metadata,
            extraction="text",
        )
    ]
