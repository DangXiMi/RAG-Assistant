# tests/unit/test_citations.py
"""Tests for turning chunk metadata into human-readable citations."""
from langchain_core.documents import Document

from src.generation.citations import (
    build_source_refs,
    label,
    location,
    source_labels,
    source_name,
)


def doc(metadata, content="chunk text", score=None):
    metadata = dict(metadata)
    if score is not None:
        metadata["score"] = score
    return Document(page_content=content, metadata=metadata)


# --- source_name ------------------------------------------------------------

def test_source_name_reads_source_key():
    assert source_name({"source": "refund_policy.pdf"}) == "refund_policy.pdf"


def test_source_name_falls_back_to_filename_keys():
    assert source_name({"filename": "a.docx"}) == "a.docx"
    assert source_name({"file_name": "b.xlsx"}) == "b.xlsx"


def test_source_name_defaults_to_unknown():
    assert source_name({}) == "unknown"
    assert source_name(None) == "unknown"


def test_source_name_prefers_source_over_filename():
    assert source_name({"source": "real.pdf", "filename": "other.pdf"}) == "real.pdf"


# --- location ---------------------------------------------------------------

def test_location_renders_page():
    assert location({"page": 3}) == "p.3"


def test_location_renders_sheet():
    assert location({"sheet": "Q2"}) == "sheet 'Q2'"


def test_location_prefers_page_over_sheet():
    assert location({"page": 2, "sheet": "Q2"}) == "p.2"


def test_location_renders_row_range_when_no_page_or_sheet():
    assert location({"row": "1-5"}) == "rows 1-5"


def test_location_empty_for_flat_documents():
    assert location({"source": "notes.txt"}) == ""


def test_location_ignores_empty_values():
    assert location({"page": None, "sheet": ""}) == ""


# --- label ------------------------------------------------------------------

def test_label_includes_file_and_page():
    assert label({"source": "report.pdf", "page": 4}) == "report.pdf, p.4"


def test_label_without_location_is_just_the_file():
    assert label({"source": "notes.txt"}) == "notes.txt"


def test_label_with_prefix():
    assert label({"source": "report.pdf", "page": 4}, prefix="2") == "[2] report.pdf, p.4"


def test_label_for_spreadsheet():
    assert label({"source": "book.xlsx", "sheet": "Costs"}) == "book.xlsx, sheet 'Costs'"


# --- build_source_refs ------------------------------------------------------

def test_build_source_refs_numbers_entries():
    docs = [
        doc({"source": "a.pdf", "page": 1}),
        doc({"source": "b.pdf", "page": 2}),
    ]

    refs = build_source_refs(docs)

    assert [r["ref"] for r in refs] == [1, 2]
    assert refs[0]["label"] == "[1] a.pdf, p.1"
    assert refs[1]["source"] == "b.pdf"
    assert refs[1]["page"] == 2


def test_build_source_refs_deduplicates_same_location():
    """Two chunks from one page must not be reported as two sources."""
    docs = [
        doc({"source": "a.pdf", "page": 1}, content="first"),
        doc({"source": "a.pdf", "page": 1}, content="second"),
    ]

    refs = build_source_refs(docs)

    assert len(refs) == 1
    assert refs[0]["location"] == "p.1"


def test_build_source_refs_keeps_different_pages():
    docs = [
        doc({"source": "a.pdf", "page": 1}),
        doc({"source": "a.pdf", "page": 2}),
    ]

    assert len(build_source_refs(docs)) == 2


def test_build_source_refs_carries_chunk_id_and_score():
    docs = [doc({"source": "a.pdf", "page": 3, "doc_id": "chunk-9"}, score=0.5)]

    ref = build_source_refs(docs)[0]

    assert ref["chunk_id"] == "chunk-9"
    assert ref["score"] == 0.5


def test_build_source_refs_accepts_plain_mappings():
    refs = build_source_refs([{"metadata": {"source": "m.pdf", "page": 7}, "score": 0.1}])

    assert refs[0]["label"] == "[1] m.pdf, p.7"


def test_build_source_refs_handles_no_docs():
    assert build_source_refs([]) == []


# --- source_labels ----------------------------------------------------------

def test_source_labels_are_plain_strings():
    docs = [
        doc({"source": "a.pdf", "page": 1}),
        doc({"source": "b.xlsx", "sheet": "Costs"}),
    ]

    labels = source_labels(docs)

    assert labels == ["[1] a.pdf, p.1", "[2] b.xlsx, sheet 'Costs'"]
    assert all(isinstance(item, str) for item in labels)


def test_source_labels_never_contains_a_bare_uuid():
    """The regression this module exists to prevent."""
    docs = [doc({"source": "a.pdf", "page": 1, "doc_id": "5c2f-uuid-here"})]

    labels = source_labels(docs)

    assert "5c2f-uuid-here" not in labels[0]
    assert "a.pdf" in labels[0]
