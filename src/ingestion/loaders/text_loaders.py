# src/ingestion/loaders/text_loaders.py
"""Loaders for flat text formats: plain text, Markdown and HTML."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.loaders.exceptions import LoaderError
from src.ingestion.loaders.interface import LoadedPage, base_metadata

logger = logging.getLogger(__name__)

# Tags whose text content should never reach the index.
_HTML_DROP_TAGS = ("script", "style", "noscript", "template")

# Block-level tags that should become line breaks so chunking keeps structure.
_HTML_BLOCK_TAGS = (
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "blockquote", "pre", "table",
)


def load_text(path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
    """Load a plain-text or Markdown file as a single citable unit."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to a lenient decode rather than failing the whole ingest.
        logger.warning(
            "UTF-8 decode failed for %s; falling back to latin-1", path.name
        )
        text = path.read_text(encoding="latin-1", errors="replace")
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    return [
        LoadedPage(
            text=text,
            index=0,
            metadata=base_metadata(path, metadata),
            extraction="text",
        )
    ]


def load_html(path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
    """Load an HTML file, keeping block structure and turning tables to Markdown."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - bs4 is a hard dependency
        raise LoaderError(
            "HTML ingestion requires the 'beautifulsoup4' package"
        ) from exc

    from src.ingestion.loaders.markdown import grid_to_markdown

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup.find_all(_HTML_DROP_TAGS):
        tag.decompose()

    # Convert every <table> to a Markdown table before flattening to text,
    # so tabular structure is not lost.
    for table in soup.find_all("table"):
        grid: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if cells:
                grid.append([cell.get_text(" ", strip=True) for cell in cells])
        markdown = grid_to_markdown(grid)
        if markdown:
            table.replace_with(soup.new_string("\n\n" + markdown + "\n\n"))
        else:
            table.decompose()

    for br in soup.find_all(_HTML_BLOCK_TAGS):
        br.insert_after(soup.new_string("\n"))

    text = soup.get_text()

    # Collapse the excess blank lines the tag insertion produced.
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)

    return [
        LoadedPage(
            text="\n".join(cleaned).strip(),
            index=0,
            metadata=base_metadata(path, metadata),
            extraction="text",
        )
    ]
