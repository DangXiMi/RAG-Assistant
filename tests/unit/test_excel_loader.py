# tests/unit/test_excel_loader.py
"""Tests for Excel and CSV loading, including sheet metadata and Markdown tables."""
import pytest

from src.ingestion.loaders import load_document
from src.ingestion.loaders.exceptions import MissingDependencyError

HAS_OPENPYXL = True
try:
    import openpyxl  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_OPENPYXL = False

requires_openpyxl = pytest.mark.skipif(
    not HAS_OPENPYXL, reason="openpyxl is not installed"
)


def build_workbook(path, sheets):
    """Create an .xlsx file from {sheet_name: rows}."""
    from openpyxl import Workbook

    workbook = Workbook()
    # Drop the default sheet so sheet order matches the input exactly.
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(title=name)
        for row in rows:
            sheet.append(row)
    workbook.save(path)
    return path


@pytest.fixture
def single_sheet(tmp_path):
    return build_workbook(
        tmp_path / "refunds.xlsx",
        {
            "Policy": [
                ["Item type", "Refund window"],
                ["Standard", "30 days"],
                ["Sale item", "Not refundable"],
            ]
        },
    )


@pytest.fixture
def multi_sheet(tmp_path):
    return build_workbook(
        tmp_path / "quarterly.xlsx",
        {
            "Revenue": [["Month", "Amount"], ["Jan", 1000], ["Feb", 1200]],
            "Costs": [["Month", "Amount"], ["Jan", 400], ["Feb", 450]],
            "Notes": [["Comment"], ["Reviewed by finance"]],
        },
    )


# --- Basic extraction -------------------------------------------------------

@requires_openpyxl
def test_loads_cell_values(single_sheet):
    pages = load_document(single_sheet)

    assert len(pages) == 1
    assert "30 days" in pages[0].text
    assert "Not refundable" in pages[0].text


@requires_openpyxl
def test_sheet_becomes_a_markdown_table(single_sheet):
    pages = load_document(single_sheet)

    assert "| Item type" in pages[0].text
    assert "|---" in pages[0].text


@requires_openpyxl
def test_sheet_name_is_in_the_metadata(single_sheet):
    pages = load_document(single_sheet)

    assert pages[0].metadata["sheet"] == "Policy"
    assert pages[0].metadata["source"] == "refunds.xlsx"


@requires_openpyxl
def test_row_range_is_recorded(single_sheet):
    pages = load_document(single_sheet)

    assert pages[0].metadata["row_start"] == 1
    assert pages[0].metadata["row_end"] == 3
    assert pages[0].metadata["row"] == "1-3"


@requires_openpyxl
def test_sheet_name_appears_in_the_text(single_sheet):
    """A chunk must stay self-describing when retrieved on its own."""
    pages = load_document(single_sheet)

    assert "Sheet: Policy" in pages[0].text


@requires_openpyxl
def test_numeric_cells_are_preserved(multi_sheet):
    pages = load_document(multi_sheet)

    assert "1000" in pages[0].text
    assert "1000.0" not in pages[0].text


# --- Multiple sheets --------------------------------------------------------

@requires_openpyxl
def test_every_sheet_becomes_a_unit(multi_sheet):
    pages = load_document(multi_sheet)

    assert len(pages) == 3
    assert [p.metadata["sheet"] for p in pages] == ["Revenue", "Costs", "Notes"]


@requires_openpyxl
def test_second_sheet_content_is_retrievable(multi_sheet):
    """The brief's core promise: data on any sheet must be findable."""
    pages = load_document(multi_sheet)

    costs = [p for p in pages if p.metadata["sheet"] == "Costs"][0]
    assert "450" in costs.text


@requires_openpyxl
def test_unit_indexes_are_sequential(multi_sheet):
    pages = load_document(multi_sheet)

    assert [p.index for p in pages] == [0, 1, 2]


@requires_openpyxl
def test_empty_sheets_are_skipped(tmp_path):
    path = build_workbook(
        tmp_path / "gappy.xlsx",
        {
            "Data": [["A"], ["1"]],
            "Empty": [[None], [None]],
        },
    )

    pages = load_document(path)

    assert [p.metadata["sheet"] for p in pages] == ["Data"]


@requires_openpyxl
def test_leading_empty_rows_and_columns_are_trimmed(tmp_path):
    path = build_workbook(
        tmp_path / "padded.xlsx",
        {"S": [[None, None, None], [None, "Header", None], [None, "value", None]]},
    )

    pages = load_document(path)

    assert "Header" in pages[0].text
    assert "value" in pages[0].text
    assert pages[0].metadata["row_start"] == 1


# --- Integration with chunking ---------------------------------------------

@requires_openpyxl
def test_chunks_carry_sheet_metadata(multi_sheet):
    from src.ingestion.chunker import chunk_pages

    pages = load_document(multi_sheet)
    chunks = chunk_pages(pages)

    assert chunks
    sheets = {c.metadata["sheet"] for c in chunks}
    assert "Revenue" in sheets

    for chunk in chunks:
        assert chunk.metadata["source"] == "quarterly.xlsx"
        assert "sheet" in chunk.metadata


# --- CSV --------------------------------------------------------------------

def test_loads_csv_as_a_markdown_table(tmp_path):
    path = tmp_path / "refunds.csv"
    path.write_text(
        "Item type,Refund window\nStandard,30 days\nSale item,Not refundable\n",
        encoding="utf-8",
    )

    pages = load_document(path)

    assert len(pages) == 1
    assert "| Item type" in pages[0].text
    assert "Not refundable" in pages[0].text


def test_csv_has_no_sheet_key(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    pages = load_document(path)

    assert "sheet" not in pages[0].metadata
    assert pages[0].metadata["source"] == "data.csv"


def test_empty_csv_produces_no_units(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    assert load_document(path) == []


# --- Missing dependency handling -------------------------------------------

def test_missing_openpyxl_message_is_actionable(tmp_path, monkeypatch):
    import src.ingestion.loaders.excel as excel_module

    monkeypatch.setattr(excel_module, "has_openpyxl", lambda: False)

    with pytest.raises(MissingDependencyError) as excinfo:
        excel_module.load_excel(tmp_path / "x.xlsx")

    assert "openpyxl" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)
