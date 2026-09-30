# tests/unit/test_chunk_pages.py
"""Tests for page-scoped chunking, which preserves citation metadata."""
import uuid

import pytest

from src.ingestion.chunker import CHUNK_SIZE, Chunk, chunk_pages, chunk_text


class FakePage:
    """Stand-in for a loader's LoadedPage."""

    def __init__(self, text, metadata, extraction="text"):
        self.text = text
        self.metadata = metadata
        self.extraction = extraction


def page(text, **metadata):
    return FakePage(text, metadata)


def test_returns_a_flat_list_of_chunks():
    pages = [page("Hello world.", source="a.pdf", page=1)]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_page_metadata_lands_on_every_chunk():
    pages = [
        page("Intro text. " * 10, source="manual.pdf", page=1),
        page("Second page. " * 10, source="manual.pdf", page=2),
    ]

    chunks = chunk_pages(pages)

    for chunk in chunks:
        assert chunk.metadata["source"] == "manual.pdf"
        assert chunk.metadata["page"] in (1, 2)


def test_each_page_keeps_its_own_page_number():
    long_text = "Sentence about refunds. " * 60  # forces several chunks
    pages = [
        page(long_text, source="doc.pdf", page=1),
        page(long_text, source="doc.pdf", page=2),
        page(long_text, source="doc.pdf", page=7),
    ]

    chunks = chunk_pages(pages)

    # Chunks are ordered, so page numbers must be non-decreasing and complete.
    assert [c.metadata["page"] for c in chunks] == sorted(
        c.metadata["page"] for c in chunks
    )
    assert {c.metadata["page"] for c in chunks} == {1, 2, 7}


def test_sheet_metadata_is_preserved():
    pages = [
        page("Sheet one data", source="book.xlsx", sheet="Revenue", row="1-5"),
        page("Sheet two data", source="book.xlsx", sheet="Costs", row="1-3"),
    ]

    chunks = chunk_pages(pages)

    assert chunks[0].metadata["sheet"] == "Revenue"
    assert chunks[0].metadata["row"] == "1-5"
    assert chunks[1].metadata["sheet"] == "Costs"


def test_chunk_index_is_global_and_sequential():
    pages = [
        page("A" * (CHUNK_SIZE + 50), source="a.pdf", page=1),
        page("B" * (CHUNK_SIZE + 50), source="a.pdf", page=2),
    ]

    chunks = chunk_pages(pages)

    indexes = [c.metadata["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks)))
    assert indexes[-1] > 1  # proves numbering continues across pages


def test_total_chunks_is_consistent_across_all_pages():
    pages = [
        page("Some text here. " * 40, source="a.pdf", page=1),
        page("Other text here. " * 40, source="a.pdf", page=2),
    ]

    chunks = chunk_pages(pages)

    for chunk in chunks:
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_document_metadata_does_not_override_page_number():
    pages = [page("Body text", source="a.pdf", page=3)]

    chunks = chunk_pages(pages, metadata={"page": 99, "collection": "kb"})

    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["collection"] == "kb"


def test_document_metadata_is_merged_when_page_lacks_the_key():
    pages = [page("Body text", source="a.pdf", page=1)]

    chunks = chunk_pages(pages, metadata={"department": "support"})

    assert chunks[0].metadata["department"] == "support"


def test_empty_pages_are_skipped():
    pages = [
        page("", source="a.pdf", page=1),
        page("   \n  ", source="a.pdf", page=2),
        page("Real content", source="a.pdf", page=3),
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 3


def test_skip_empty_false_keeps_blank_units():
    pages = [page("", source="a.pdf", page=1)]

    chunks = chunk_pages(pages, skip_empty=False)

    assert len(chunks) == 1


def test_all_empty_pages_returns_no_chunks():
    pages = [page("", source="a.pdf", page=1), page("  ", source="a.pdf", page=2)]

    assert chunk_pages(pages) == []


def test_accepts_mapping_pages_as_well_as_objects():
    pages = [{"text": "Mapped page text", "metadata": {"source": "m.pdf", "page": 4}}]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "m.pdf"
    assert chunks[0].metadata["page"] == 4


def test_offsets_stay_within_their_own_page():
    text = "Grounding sentence here. " * 60
    pages = [page(text, source="a.pdf", page=1), page(text, source="a.pdf", page=2)]

    chunks = chunk_pages(pages)

    for chunk in chunks:
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)


def test_chunk_ids_are_unique_across_pages():
    pages = [page("text " * 200, source="a.pdf", page=i) for i in range(1, 4)]

    chunks = chunk_pages(pages)
    ids = [str(c.id) for c in chunks]

    assert len(ids) == len(set(ids))
    for chunk_id in ids:
        uuid.UUID(chunk_id)


def test_chunk_text_contract_is_unchanged():
    """The original single-string API must keep behaving as before."""
    text = "This is a short document."
    chunks = chunk_text(text, {"source": "test"})

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == 1


def test_chunk_text_empty_string_still_returns_one_chunk():
    chunks = chunk_text("", {"source": "empty"})

    assert len(chunks) == 1
    assert chunks[0].text == ""
