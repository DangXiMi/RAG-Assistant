# tests/unit/test_ocr_loader.py
"""Tests for the OCR path used by images and scanned PDFs.

Tests that need a working OCR backend are skipped when the Tesseract engine
is not installed, so the suite stays green on machines without it. Tests that
assert on *graceful degradation* always run, because that behaviour must hold
regardless of what is installed.
"""
import pytest

from src.ingestion.loaders import load_document
from src.ingestion.loaders.capabilities import (
    has_pytesseract,
    has_tesseract_binary,
    ocr_available,
    ocr_unavailable_reason,
)
from src.ingestion.loaders.exceptions import OcrNotAvailableError

OCR_READY = ocr_available()

requires_ocr = pytest.mark.skipif(
    not OCR_READY,
    reason=f"OCR backend unavailable: {ocr_unavailable_reason()}",
)


def build_text_image(path, text):
    """Render text into a PNG large enough for Tesseract to read reliably."""
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")

    image = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), text, fill="black", font_size=48)
    image.save(path)
    return path


# --- Capability detection (always runs) -------------------------------------

def test_capability_probe_returns_booleans():
    assert isinstance(has_pytesseract(), bool)
    assert isinstance(has_tesseract_binary(), bool)
    assert isinstance(ocr_available(), bool)


def test_ocr_requires_both_wrapper_and_binary():
    # Usable OCR means the Python wrapper *and* the engine binary are present.
    assert ocr_available() == (has_pytesseract() and has_tesseract_binary())


def test_unavailable_reason_is_a_non_empty_string():
    assert isinstance(ocr_unavailable_reason(), str)
    assert ocr_unavailable_reason()


def test_capability_summary_lists_all_backends():
    from src.ingestion.loaders.capabilities import summary

    snapshot = summary()

    assert "ocr (pytesseract)" in snapshot
    assert "ocr (tesseract binary)" in snapshot
    assert "ocr (usable)" in snapshot


# --- Graceful degradation (always runs) -------------------------------------

def test_image_ingestion_reports_missing_ocr_clearly(tmp_path):
    """Without OCR, an image upload must fail with install instructions."""
    path = tmp_path / "scan.png"
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (100, 100), "white").save(path)

    if OCR_READY:
        # Backend present: it should simply succeed instead.
        pages = load_document(path)
        assert len(pages) == 1
        return

    with pytest.raises(OcrNotAvailableError) as excinfo:
        load_document(path)

    message = str(excinfo.value)
    assert "OCR" in message
    # The message must tell the user how to fix it.
    assert "install" in message.lower()


def test_ensure_ocr_ready_matches_availability():
    from src.ingestion.loaders.ocr import ensure_ocr_ready

    if OCR_READY:
        ensure_ocr_ready()  # must not raise
    else:
        with pytest.raises(OcrNotAvailableError):
            ensure_ocr_ready()


def test_describe_backend_reports_status():
    from src.ingestion.loaders.ocr import describe_backend

    description = describe_backend()

    assert isinstance(description, str)
    assert description
    if OCR_READY:
        assert "tesseract" in description.lower()


def test_image_extensions_are_routed_to_the_image_loader():
    from src.ingestion.loaders import LOADER_REGISTRY

    for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        assert LOADER_REGISTRY[extension] == "image"


# --- Real OCR (only when a backend is installed) ----------------------------

@requires_ocr
def test_ocr_extracts_text_from_an_image(tmp_path):
    path = build_text_image(tmp_path / "note.png", "REFUND POLICY 30 DAYS")

    pages = load_document(path)

    assert len(pages) == 1
    found = pages[0].text.upper()
    assert "REFUND" in found
    assert "30" in found


@requires_ocr
def test_image_unit_is_marked_as_ocr(tmp_path):
    path = build_text_image(tmp_path / "note.png", "HELLO WORLD")

    pages = load_document(path)

    assert pages[0].extraction == "ocr"
    assert pages[0].metadata["ocr"] is True


@requires_ocr
def test_image_unit_has_page_one_for_citation(tmp_path):
    path = build_text_image(tmp_path / "note.png", "CITE ME")

    pages = load_document(path)

    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["source"] == "note.png"
    assert pages[0].label == "note.png, p.1"


@requires_ocr
def test_ocr_result_is_chunkable(tmp_path):
    from src.ingestion.chunker import chunk_pages

    path = build_text_image(tmp_path / "long.png", "REFUND POLICY DETAILS" * 20)

    pages = load_document(path)
    chunks = chunk_pages(pages)

    assert chunks
    assert all(c.metadata["source"] == "long.png" for c in chunks)


@requires_ocr
def test_scanned_pdf_is_ocrd_end_to_end(tmp_path):
    """A rasterised text page inside a PDF must be recovered via OCR."""
    pymupdf = pytest.importorskip("pymupdf")
    from PIL import Image, ImageDraw

    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(image).text((20, 80), "SCANNED REFUND POLICY", fill="black", font_size=48)
    image.save(image_path)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 900, 200), filename=str(image_path))
    pdf_path = tmp_path / "scanned.pdf"
    document.save(str(pdf_path))
    document.close()

    pages = load_document(pdf_path)

    assert len(pages) == 1
    assert pages[0].extraction == "ocr"
    assert "REFUND" in pages[0].text.upper()
    assert pages[0].metadata["page"] == 1
