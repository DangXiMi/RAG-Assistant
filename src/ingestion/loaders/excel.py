# src/ingestion/loaders/excel.py
"""Loaders for spreadsheets: .xlsx, .xlsm and .csv.

Each worksheet becomes one `LoadedPage` whose text is a Markdown table, so
row/column structure survives chunking. Metadata records the sheet name and
the row range the content came from.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.ingestion.loaders.capabilities import has_openpyxl
from src.ingestion.loaders.exceptions import LoaderError, MissingDependencyError
from src.ingestion.loaders.interface import LoadedPage, base_metadata
from src.ingestion.loaders.markdown import grid_to_markdown

logger = logging.getLogger(__name__)

OPENPYXL_HINT = "Install it with: pip install openpyxl"

# Rows scanned when guessing whether a sheet actually has a header row.
_HEADER_PROBE_ROWS = 5

# Below this many populated cells a row is treated as noise and dropped.
_MIN_CELLS_PER_ROW = 1


def _row_is_empty(row: Sequence[Any]) -> bool:
    """True when a row holds no meaningful values."""
    for cell in row:
        if cell is None:
            continue
        if isinstance(cell, str) and not cell.strip():
            continue
        return False
    return True


def _looks_like_header(row: Sequence[Any]) -> bool:
    """Heuristic: a header row is non-empty text without long numeric values."""
    values = [c for c in row if c is not None and str(c).strip()]
    if not values:
        return False
    texty = sum(1 for v in values if isinstance(v, str) and not _is_number(v))
    return texty >= max(1, len(values) // 2)


def _is_number(value: str) -> bool:
    """True when the string parses as a number (so it is data, not a heading)."""
    try:
        float(value.replace(",", "").replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def trim_grid(grid: list[list[Any]]) -> list[list[Any]]:
    """Drop leading/trailing empty rows and pad rows to a rectangle."""
    while grid and _row_is_empty(grid[0]):
        grid.pop(0)
    while grid and _row_is_empty(grid[-1]):
        grid.pop()

    if not grid:
        return []

    width = max(len(row) for row in grid)
    # Excel often reports trailing empty columns; trim those too.
    while width > 0 and all(
        len(row) < width or row[width - 1] is None or str(row[width - 1]).strip() == ""
        for row in grid
    ):
        width -= 1

    if width == 0:
        return []

    return [
        list(row[:width]) + [None] * (width - len(row[:width]))
        for row in grid
        if not _row_is_empty(row)
    ]


def sheet_to_page(
    grid: list[list[Any]],
    sheet_name: str,
    path: Path,
    metadata: dict[str, Any] | None,
    unit_index: int,
) -> LoadedPage | None:
    """Render one worksheet's grid as a `LoadedPage`, or None if it is empty."""
    grid = trim_grid(grid)
    if not grid:
        return None

    header = grid[0] if _looks_like_header(grid[0]) else None
    data_rows = grid[1:] if header else grid

    table = grid_to_markdown(grid)
    if not table:
        return None

    # Repeat the sheet name in the text so a chunk citing this sheet is
    # self-describing even when retrieved in isolation.
    text = f"Sheet: {sheet_name}\n\n{table}"

    row_start = 1
    row_end = len(grid)
    page_metadata = {
        **base_metadata(path, metadata),
        "sheet": sheet_name,
        "row_start": row_start,
        "row_end": row_end,
        "row": f"{row_start}-{row_end}",
    }

    return LoadedPage(
        text=text,
        index=unit_index,
        metadata=page_metadata,
        extraction="text",
        # Keep a handle on the parsed table for tests and debugging.
        char_count=len(text),
    )


def load_excel(path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
    """Load an .xlsx/.xlsm workbook, one `LoadedPage` per non-empty sheet."""
    if not has_openpyxl():
        raise MissingDependencyError("Excel ingestion", "openpyxl", OPENPYXL_HINT)

    from openpyxl import load_workbook

    try:
        # data_only=True returns cached formula results instead of "=A1+B1".
        workbook = load_workbook(filename=path, data_only=True, read_only=True)
    except Exception as exc:
        raise LoaderError(f"Could not open spreadsheet {path.name}: {exc}") from exc

    pages: list[LoadedPage] = []
    try:
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows = [list(row) for row in sheet.iter_rows(values_only=True)]
            page = sheet_to_page(rows, sheet_name, path, metadata, len(pages))
            if page is not None:
                pages.append(page)
            else:
                logger.info("Skipping empty sheet '%s' in %s", sheet_name, path.name)
    finally:
        workbook.close()

    if not pages:
        logger.warning("No usable content found in spreadsheet %s", path.name)

    return pages


def load_csv(path: Path, metadata: dict[str, Any] | None = None) -> list[LoadedPage]:
    """Load a CSV file as a single Markdown-table unit."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            grid: list[list[Any]] = [row for row in csv.reader(handle)]
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", newline="") as handle:
            grid = [row for row in csv.reader(handle)]
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    page = sheet_to_page(grid, "csv", path, metadata, 0)
    if page is None:
        return []

    # CSV has no sheet concept; label the location by row range instead.
    page.metadata.pop("sheet", None)
    page.metadata.pop("row_start", None)
    page.metadata.pop("row_end", None)
    page.metadata["location"] = page.metadata.pop("row", "")

    return [page]
