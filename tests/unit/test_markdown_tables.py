# tests/unit/test_markdown_tables.py
"""Tests for the Markdown table renderer used by Excel, docx and PDF loaders."""
from src.ingestion.loaders.markdown import grid_to_markdown, rows_to_markdown


def test_grid_uses_first_row_as_header():
    markdown = grid_to_markdown([["Name", "Age"], ["Alice", 30], ["Bob", 25]])

    lines = markdown.splitlines()
    assert len(lines) == 4
    assert "Name" in lines[0] and "Age" in lines[0]
    assert set(lines[1]) <= set("-| ")
    assert "Alice" in lines[2]
    assert "Bob" in lines[3]


def test_empty_grid_returns_empty_string():
    assert grid_to_markdown([]) == ""
    assert rows_to_markdown([]) == ""


def test_all_empty_rows_returns_empty_string():
    assert grid_to_markdown([[None, None], ["", "  "]]) == ""


def test_header_only_grid_still_produces_a_table():
    markdown = grid_to_markdown([["Name", "Age"]])

    assert "Name" in markdown
    assert "---" in markdown


def test_one_column_grid_is_not_treated_as_a_separator_row():
    # A single row of dashes must not be mistaken for a separator row.
    markdown = grid_to_markdown([["A", "B"]])

    assert markdown.count("|") >= 6


def test_pipes_in_cells_are_escaped():
    markdown = grid_to_markdown([["Expr", "Result"], ["a | b", "ok"]])

    assert r"a \| b" in markdown
    # The escaped pipe must not add a column.
    data_line = markdown.splitlines()[2]
    assert data_line.count(" | ") == 1


def test_newlines_in_cells_are_flattened():
    markdown = grid_to_markdown([["Notes"], ["line one\nline two"]])

    assert "line one line two" in markdown
    assert markdown.count("\n") == 2


def test_none_and_numeric_cells_are_rendered():
    markdown = grid_to_markdown([["Item", "Qty"], ["Widget", 5], ["Gadget", None]])

    assert "Widget" in markdown
    assert "5" in markdown
    assert "None" not in markdown


def test_float_integers_lose_the_decimal_point():
    markdown = grid_to_markdown([["Qty"], [2.0]])

    assert " 2 " in markdown
    assert "2.0" not in markdown


def test_ragged_rows_are_padded():
    markdown = grid_to_markdown([["A", "B", "C"], ["1"], ["2", "3"]])

    data_lines = [line for line in markdown.splitlines()[2:] if line.strip()]
    widths = {line.count("|") for line in data_lines}
    assert len(widths) == 1


def test_columns_are_aligned():
    markdown = grid_to_markdown([["Short", "LongerHeader"], ["x", "y"]])

    lines = markdown.splitlines()
    assert len(lines[0]) == len(lines[2])


def test_text_content_is_preserved_for_retrieval():
    grid = [
        ["Policy", "Refund window"],
        ["Standard refund", "30 days from delivery"],
        ["Sale items", "Not refundable"],
    ]

    markdown = grid_to_markdown(grid)

    assert "Refund window" in markdown
    assert "30 days from delivery" in markdown
    assert "Not refundable" in markdown
