# tests/unit/test_pdf_loader.py
"""Tests for PDF loading: per-page text, page metadata and scanned-page OCR."""
import pytest

from src.ingestion.loaders import load_document

HAS_PYMUPDF = True
try:
    import pymupdf  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_PYMUPDF = False

requires_pymupdf = pytest.mark.skipif(
    not HAS_PYMUPDF, reason="pymupdf is not installed"
)


def build_text_pdf(path, pages):
    """Write a PDF with a real text layer, one string per page."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 100), text, fontsize=12)
    document.save(str(path))
    document.close()
    return path


def build_scanned_pdf(path, page_count=1):
    """Write a PDF with no text layer at all, emulating a scan.

    Only image content is added, so native extraction yields nothing and the
    loader must fall back to OCR.
    """
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(page_count):
        page = document.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200))
        pixmap.set_rect(pixmap.irect, (255, 255, 255))
        page.insert_image(pymupdf.Rect(0, 0, 400, 400), pixmap=pixmap)
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def text_pdf(tmp_path):
    return build_text_pdf(
        tmp_path / "policy.pdf",
        [
            "Refund Policy: refunds are processed within 30 days.",
            "Exceptions: sale items are not refundable.",
        ],
    )


@pytest.fixture
def single_page_pdf(tmp_path):
    return build_text_pdf(tmp_path / "one.pdf", ["Only page content here."])


# --- Text-layer PDFs --------------------------------------------------------

@requires_pymupdf
def test_one_unit_per_page(text_pdf):
    pages = load_document(text_pdf)

    assert len(pages) == 2


@requires_pymupdf
def test_page_numbers_are_one_based(text_pdf):
    pages = load_document(text_pdf)

    assert [p.metadata["page"] for p in pages] == [1, 2]


@requires_pymupdf
def test_page_count_is_recorded(text_pdf):
    pages = load_document(text_pdf)

    for page in pages:
        assert page.metadata["page_count"] == 2


@requires_pymupdf
def test_text_is_extracted_per_page(text_pdf):
    pages = load_document(text_pdf)

    assert "30 days" in pages[0].text
    assert "not refundable" in pages[1].text
    # Page 1 must not contain page 2's content.
    assert "not refundable" not in pages[0].text


@requires_pymupdf
def test_extraction_method_is_native_for_text_pdfs(text_pdf):
    pages = load_document(text_pdf)

    assert all(p.extraction == "native" for p in pages)


@requires_pymupdf
def test_source_filename_is_preserved(text_pdf):
    pages = load_document(text_pdf)

    assert all(p.metadata["source"] == "policy.pdf" for p in pages)


@requires_pymupdf
def test_chunks_cite_the_right_page(text_pdf):
    from src.ingestion.chunker import chunk_pages

    pages = load_document(text_pdf)
    chunks = chunk_pages(pages)

    by_page = {c.metadata["page"]: c.text for c in chunks}
    assert "30 days" in by_page[1]
    assert "not refundable" in by_page[2]


# --- Fallback to pypdf ------------------------------------------------------

@requires_pymupdf
def test_falls_back_to_pypdf_when_pymupdf_is_unavailable(text_pdf, monkeypatch):
    import src.ingestion.loaders.pdf as pdf_module

    monkeypatch.setattr(pdf_module, "_native_text_pymupdf", lambda path: None)

    pages = load_document(text_pdf)

    assert len(pages) == 2
    assert "30 days" in pages[0].text


# --- Scanned PDFs -----------------------------------------------------------

@requires_pymupdf
def test_scanned_pdf_without_ocr_raises_actionable_error(tmp_path):
    """A scan with OCR disabled must fail loudly, not silently index nothing."""
    from src.ingestion.loaders.exceptions import ScannedDocumentError

    path = build_scanned_pdf(tmp_path / "scan.pdf")

    with pytest.raises(ScannedDocumentError) as excinfo:
        load_document(path, ocr=False)

    assert "scan.pdf" in str(excinfo.value)


@requires_pymupdf
def test_scanned_pdf_attempts_ocr_when_enabled(tmp_path, monkeypatch):
    """With OCR enabled the page must be routed to the OCR backend."""
    import src.ingestion.loaders.pdf as pdf_module

    calls = {"count": 0}

    def fake_ocr(page, language="eng", dpi=300):
        calls["count"] += 1
        return "Recognised text from the scanned page."

    monkeypatch.setattr(pdf_module, "ocr_pdf_page", fake_ocr)
    monkeypatch.setattr(pdf_module, "ocr_available", lambda: True)

    path = build_scanned_pdf(tmp_path / "scan.pdf")
    pages = load_document(path, ocr=True)

    assert calls["count"] == 1
    assert "Recognised text" in pages[0].text
    assert pages[0].extraction == "ocr"


@requires_pymupdf
def test_native_text_wins_when_it_is_richer(tmp_path, monkeypatch):
    """OCR output must not replace a perfectly good text layer."""
    import src.ingestion.loaders.pdf as pdf_module

    monkeypatch.setattr(pdf_module, "ocr_pdf_page", lambda *a, **k: "noise")
    monkeypatch.setattr(pdf_module, "ocr_available", lambda: True)

    path = build_text_pdf(tmp_path / "digital.pdf", ["A" * 200])
    pages = load_document(path, ocr=True)

    assert pages[0].extraction == "native"
    assert "A" * 50 in pages[0].text


@requires_pymupdf
def test_ocr_is_skipped_for_pages_with_enough_text(tmp_path, monkeypatch):
    """Text pages must never pay the OCR cost.

    The page text must exceed `min_text_chars`, otherwise the loader correctly
    treats it as a scan and OCR is attempted.
    """
    import src.ingestion.loaders.pdf as pdf_module

    def explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("OCR should not run on a text PDF")

    monkeypatch.setattr(pdf_module, "ocr_pdf_page", explode)

    path = build_text_pdf(
        tmp_path / "digital.pdf",
        ["Refunds are processed within 30 days of delivery. " * 3],
    )

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].extraction == "native"


@requires_pymupdf
def test_single_page_pdf_has_no_second_page(single_page_pdf):
    pages = load_document(single_page_pdf)

    assert len(pages) == 1
    assert pages[0].metadata["page"] == 1
