# tests/unit/test_load_pages_entrypoint.py
"""Tests for `load_pages`, the entry point the ingestion worker actually calls.

Loader tests exercise `load_document` directly, so they cannot catch wiring
mistakes between the loader and the cleaning stage. These tests close that gap:
an option-plumbing bug here failed every ingest at runtime while the rest of the
suite stayed green.
"""
import pytest

from src.utils.helper_func import extract_text, load_pages


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text(
        "Refund Policy\n\nRefunds are processed within 30 days.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def messy_pages(tmp_path):
    """A document whose text needs cleaning."""
    path = tmp_path / "messy.txt"
    path.write_text(
        "Body text with    excess   spaces\nPage 1 of 2\nmore text",
        encoding="utf-8",
    )
    return path


def test_load_pages_returns_units(text_file):
    pages = load_pages(text_file)

    assert len(pages) >= 1
    assert "30 days" in pages[0].text


def test_load_pages_with_default_config_does_not_raise(text_file):
    """Regression: cleaning options must not be forwarded to the loaders.

    `load_document` does not accept `fix_hyphenation`/`remove_furniture`, so
    passing the merged option dict straight through raised TypeError and broke
    every ingestion.
    """
    pages = load_pages(text_file)

    assert pages


def test_load_pages_applies_cleaning_by_default(messy_pages):
    pages = load_pages(messy_pages)

    assert "   " not in pages[0].text
    assert "Page 1 of 2" not in pages[0].text


def test_cleaning_can_be_disabled(text_file):
    pages = load_pages(text_file, clean=False)

    assert pages


def test_cleaning_options_are_forwarded(text_file):
    """Explicit cleaning flags must reach `clean_pages`, not the loader."""
    pages = load_pages(
        text_file,
        clean=True,
        remove_furniture=False,
        fix_hyphenation=False,
        strip_page_numbers=False,
    )

    assert pages


def test_extract_text_round_trips_through_load_pages(text_file):
    text = extract_text(text_file)

    assert "30 days" in text


def test_load_pages_preserves_page_metadata(tmp_path):
    """Requires pymupdf to build a multi-page PDF."""
    pymupdf = pytest.importorskip("pymupdf")

    path = tmp_path / "two_pages.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 100), "Refunds are processed within 30 days.", fontsize=12)
    second = document.new_page()
    second.insert_text((72, 100), "Sale items are not refundable.", fontsize=12)
    document.save(str(path))
    document.close()

    pages = load_pages(path)

    assert len(pages) == 2
    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert all(p.metadata["source"] == "two_pages.pdf" for p in pages)


def test_load_pages_keeps_sheet_metadata(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    sheet = workbook.create_sheet("Costs")
    sheet.append(["Region", "Lead time"])
    sheet.append(["Europe", "5 business days"])
    workbook.save(path)

    pages = load_pages(path)

    assert pages[0].metadata["sheet"] == "Costs"
    assert "5 business days" in pages[0].text


def test_load_pages_and_chunk_preserve_citations(tmp_path):
    """The full worker path: load -> clean -> chunk, metadata intact."""
    from src.ingestion.chunker import chunk_pages

    path = tmp_path / "notes.txt"
    path.write_text(
        "Refund Policy\nRefunds are processed within 30 days.\nPage 1 of 1\n",
        encoding="utf-8",
    )

    chunks = chunk_pages(load_pages(path))

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "notes.txt"
        assert "Page 1 of 1" not in chunk.text
