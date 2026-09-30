# src/ingestion/loaders/markdown.py
"""Render tabular data as Markdown.

Tables are preserved as Markdown so column/row structure survives chunking
and stays retrievable, instead of being flattened into an unsearchable blob.
"""
from __future__ import annotations

from typing import Iterable, Sequence

CELL_SEPARATOR = " | "


def _normalize_cell(value: object) -> str:
    """Flatten a single cell to a one-line string safe for a Markdown table."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        # 2.0 -> "2", which is what a spreadsheet user expects to read.
        return str(int(value))
    text = str(value)
    # Pipes and newlines would break table structure.
    text = text.replace("|", r"\|")
    text = " ".join(text.split())
    return text.strip()


def _column_widths(rows: Sequence[Sequence[str]]) -> list[int]:
    """Widest cell per column, used to pad the table into aligned columns."""
    width = 0
    for row in rows:
        width = max(width, len(row))
    widths = [0] * width
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def _is_separator_row(row: Sequence[str]) -> bool:
    """Markdown separator rows are made only of dashes, colons and spaces."""
    return all(cell and set(cell) <= set("-: ") for cell in row)


def rows_to_markdown(
    rows: Iterable[Sequence[object]],
    header: Sequence[object] | None = None,
) -> str:
    """Convert rows into a Markdown table.

    Args:
        rows: Data rows. Cells may be any type; values are normalised.
        header: Optional header row. When omitted the first data row is used
            as the header, since Markdown tables require one.

    Returns:
        A Markdown table string, or `""` when there is no usable data.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_normalize_cell(cell) for cell in row]
        if any(cells):
            cleaned.append(cells)

    if header is not None:
        head = [_normalize_cell(cell) for cell in header]
        if any(head):
            cleaned.insert(0, head)

    if not cleaned:
        return ""

    # A header is mandatory in Markdown. Promote the first row if needed.
    if len(cleaned) == 1 or _is_separator_row(cleaned[0]):
        cleaned.insert(0, [f"col{i + 1}" for i in range(len(cleaned[0]))])

    widths = _column_widths(cleaned)
    width_count = len(widths)

    def format_row(row: Sequence[str]) -> str:
        padded = list(row) + [""] * (width_count - len(row))
        return "| " + CELL_SEPARATOR.join(
            cell.ljust(widths[i]) for i, cell in enumerate(padded)
        ) + " |"

    lines = [format_row(cleaned[0])]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    lines.extend(format_row(row) for row in cleaned[1:])
    return "\n".join(lines)


def grid_to_markdown(grid: Sequence[Sequence[object]]) -> str:
    """Convert a rectangular grid (list of rows) into a Markdown table.

    Convenience wrapper around `rows_to_markdown` that treats the first row
    as the header, which is the convention for spreadsheets and PDF tables.
    """
    if not grid:
        return ""
    first, *rest = grid
    return rows_to_markdown(rest, header=first)
