# tests/unit/test_loaders.py
"""Tests for the multi-format loader registry and the flat-format loaders."""
import pytest

from src.ingestion.loaders import (
    LOADER_REGISTRY,
    OCR_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    base_metadata,
    is_supported,
    load_document,
    supported_extension_list,
)
from src.ingestion.loaders.exceptions import (
    LoaderError,
    UnsupportedFormatError,
)


# --- Registry / dispatch ----------------------------------------------------

def test_required_formats_are_registered():
    """The assignment mandates PDF, Word, Excel, images and scanned PDFs."""
    for extension in (".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".html", ".png", ".jpg"):
        assert extension in LOADER_REGISTRY, f"{extension} has no loader"


def test_image_extensions_are_flagged_as_ocr():
    assert ".png" in OCR_EXTENSIONS
    assert ".jpg" in OCR_EXTENSIONS
    assert ".pdf" not in OCR_EXTENSIONS  # only scanned pages need OCR


def test_supported_extension_list_has_no_dots_by_default():
    extensions = supported_extension_list()

    assert "pdf" in extensions
    assert ".pdf" not in extensions
    assert all(not ext.startswith(".") for ext in extensions)


def test_supported_extension_list_with_dots_when_requested():
    extensions = supported_extension_list(include_dot=True)

    assert ".pdf" in extensions


def test_is_supported_is_case_insensitive():
    assert is_supported("REPORT.PDF")
    assert is_supported("Sheet.XLSX")


def test_is_supported_rejects_unknown_extensions():
    assert not is_supported("archive.zip")
    assert not is_supported("script.exe")


def test_unsupported_format_raises_with_helpful_message(tmp_path):
    path = tmp_path / "notes.rtf"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError) as excinfo:
        load_document(path)

    assert ".rtf" in str(excinfo.value)
    assert ".pdf" in str(excinfo.value)


def test_missing_file_raises_loader_error(tmp_path):
    with pytest.raises(LoaderError):
        load_document(tmp_path / "does_not_exist.pdf")


# --- base_metadata ----------------------------------------------------------

def test_base_metadata_uses_filename_as_source(tmp_path):
    path = tmp_path / "policy.pdf"

    meta = base_metadata(path)

    assert meta["source"] == "policy.pdf"


def test_base_metadata_prefers_original_upload_name(tmp_path):
    """Uploads are stored as <job_id>_<name>; citations must use the real name."""
    path = tmp_path / "a1b2c3d4_refund_policy.pdf"

    meta = base_metadata(path, {"original_filename": "refund_policy.pdf"})

    assert meta["source"] == "refund_policy.pdf"


def test_base_metadata_ignores_caller_attempt_to_set_source(tmp_path):
    path = tmp_path / "real.pdf"

    meta = base_metadata(path, {"source": "spoofed.pdf"})

    assert meta["source"] == "real.pdf"


# --- Plain text -------------------------------------------------------------

def test_load_text_returns_one_unit(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Refunds are processed within 30 days.", encoding="utf-8")

    pages = load_document(path)

    assert len(pages) == 1
    assert "30 days" in pages[0].text
    assert pages[0].metadata["source"] == "notes.txt"
    assert pages[0].extraction == "text"


def test_load_text_handles_utf8_content(tmp_path):
    path = tmp_path / "unicode.txt"
    path.write_text("Hoàn tiền trong 30 ngày. こんにちは。", encoding="utf-8")

    pages = load_document(path)

    assert "Hoàn tiền" in pages[0].text
    assert "こんにちは" in pages[0].text


def test_load_text_falls_back_on_invalid_utf8(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9 policy")

    pages = load_document(path)

    assert "policy" in pages[0].text


def test_load_markdown_is_supported(tmp_path):
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")

    pages = load_document(path)

    assert "# Title" in pages[0].text


# --- HTML -------------------------------------------------------------------

def test_load_html_keeps_visible_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><h1>Refund Policy</h1><p>30 days.</p></body></html>",
        encoding="utf-8",
    )

    pages = load_document(path)

    assert "Refund Policy" in pages[0].text
    assert "30 days." in pages[0].text


def test_load_html_drops_script_and_style(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>.x{color:red}</style></head>"
        "<body><script>alert('ignore me')</script><p>Real content</p></body></html>",
        encoding="utf-8",
    )

    pages = load_document(path)

    assert "Real content" in pages[0].text
    assert "alert" not in pages[0].text
    assert "color:red" not in pages[0].text


def test_load_html_converts_tables_to_markdown(tmp_path):
    path = tmp_path / "table.html"
    path.write_text(
        "<html><body><table>"
        "<tr><th>Item</th><th>Window</th></tr>"
        "<tr><td>Standard</td><td>30 days</td></tr>"
        "</table></body></html>",
        encoding="utf-8",
    )

    pages = load_document(path)

    assert "| Item" in pages[0].text
    assert "30 days" in pages[0].text
    assert "|---" in pages[0].text


# --- LoadedPage behaviour ---------------------------------------------------

def test_page_label_includes_page_number(tmp_path):
    from src.ingestion.loaders.interface import LoadedPage

    loaded = LoadedPage(text="x", metadata={"source": "doc.pdf", "page": 3})

    assert loaded.label == "doc.pdf, p.3"
    assert loaded.location() == "p.3"


def test_page_label_includes_sheet_name():
    from src.ingestion.loaders.interface import LoadedPage

    loaded = LoadedPage(text="x", metadata={"source": "book.xlsx", "sheet": "Q2"})

    assert loaded.label == "book.xlsx, sheet 'Q2'"


def test_char_count_is_computed_automatically():
    from src.ingestion.loaders.interface import LoadedPage

    loaded = LoadedPage(text="12345")

    assert loaded.char_count == 5
    assert not loaded.is_empty


def test_empty_page_is_detected():
    from src.ingestion.loaders.interface import LoadedPage

    assert LoadedPage(text="   \n ").is_empty
    assert LoadedPage(text="").is_empty
