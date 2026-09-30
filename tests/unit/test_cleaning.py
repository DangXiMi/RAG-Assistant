# tests/unit/test_cleaning.py
"""Tests for the text-cleaning stage between loading and chunking."""
from src.ingestion.cleaning import (
    clean_page,
    clean_pages,
    clean_text,
    find_repeated_lines,
    remove_repeated_lines,
)


class Page:
    def __init__(self, text, **metadata):
        self.text = text
        self.metadata = metadata
        self.char_count = len(text)
        self.extraction = "text"

    @property
    def is_empty(self):
        return not (self.text or "").strip()


# --- clean_text basics ------------------------------------------------------

def test_empty_input_is_safe():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_collapses_runs_of_spaces():
    assert clean_text("too    many     spaces") == "too many spaces"


def test_collapses_excess_blank_lines():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("\n\n   hello   \n\n") == "hello"


def test_removes_control_characters():
    cleaned = clean_text("bad\x00chars\x07here")

    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "badcharshere" == cleaned


def test_normalises_non_breaking_spaces():
    assert clean_text("a\u00a0b") == "a b"


def test_removes_zero_width_characters():
    assert clean_text("a\u200bb\ufeffc") == "abc"


def test_removes_soft_hyphen():
    assert clean_text("co\u00adoperate") == "cooperate"


def test_normalises_crlf_line_endings():
    assert clean_text("a\r\nb") == "a\nb"
    assert clean_text("a\rb") == "a\nb"


def test_unicode_content_is_preserved():
    text = "Hoàn tiền trong 30 ngày. こんにちは。"

    assert clean_text(text) == text


# --- hyphenation ------------------------------------------------------------

def test_rejoins_hyphenated_line_break():
    assert clean_text("informa-\ntion") == "information"


def test_hyphenation_repair_can_be_disabled():
    assert clean_text("informa-\ntion", fix_hyphenation=False) == "informa-\ntion"


def test_leaves_capitalised_hyphenation_alone():
    """A compound like "State-\nOf" must not be welded together."""
    assert clean_text("State-\nOf-the-Art") == "State-\nOf-the-Art"


def test_leaves_genuine_hyphens_alone():
    assert clean_text("well-known fact") == "well-known fact"


# --- page numbers -----------------------------------------------------------

def test_strips_an_explicitly_labelled_page_number_line():
    assert clean_text("Chapter one\nPage 12\nMore text") == "Chapter one\nMore text"


def test_keeps_a_bare_number_without_document_context():
    """A lone number may be content (a year), so one page cannot prove it is a
    page number. Document-level detection handles real pagination instead."""
    assert clean_text("Chapter one\n12\nMore text") == "Chapter one\n12\nMore text"


def test_strips_decorated_page_numbers():
    for variant in ("Page 12", "12 of 40", "Page 12 of 40"):
        cleaned = clean_text(f"Body text\n{variant}\nMore body")
        assert variant not in cleaned, f"failed to strip {variant!r}"


def test_bare_pagination_is_removed_across_pages():
    """When numbers ascend 1,2,3 they are pagination and do get dropped."""
    pages = [Page(f"Body of page {i}\n{i}", page=i) for i in range(1, 5)]

    cleaned = clean_pages(pages)

    for page in cleaned:
        assert page.text == f"Body of page {page.metadata['page']}"


def test_does_not_strip_numbers_inside_sentences():
    text = "The mirror is 6.5 meters across."

    assert clean_text(text) == text


def test_does_not_strip_a_year_on_its_own_line():
    """A standalone year is content; only plausible page numbers are dropped."""
    assert "1990" in clean_text("Hubble launched\n1990\nDetails follow")


def test_page_number_stripping_can_be_disabled():
    assert "12" in clean_text("Text\n12\nMore", strip_page_numbers=False)


# --- repeated furniture -----------------------------------------------------

def page_with_furniture(body, number):
    return (
        f"JWST Handbook\n"
        f"{body}\n"
        f"Page {number} of 5\n"
        f"Confidential"
    )


def test_detects_repeated_header_and_footer():
    pages = [
        page_with_furniture(f"Unique body {i}", i) for i in range(1, 6)
    ]

    repeated = find_repeated_lines(pages)

    assert "jwst handbook" in repeated
    assert "confidential" in repeated


def test_does_not_flag_lines_appearing_on_too_few_pages():
    pages = [
        "Header\nBody one\nFooter",
        "Header\nBody two\nFooter",
        "Different\nBody three\nOther",
    ]

    repeated = find_repeated_lines(pages)

    # 2 of 3 pages is below the 60% threshold.
    assert "header" not in repeated


def test_single_page_documents_never_drop_lines():
    """With one page there is no pattern, and dropping could remove content."""
    assert find_repeated_lines(["Header\nReal content\nFooter"]) == set()


def test_two_page_documents_never_drop_lines():
    assert find_repeated_lines(["Header\nA\nFooter", "Header\nB\nFooter"]) == set()


def test_prose_lines_are_not_treated_as_furniture():
    """A repeated *sentence* is content; only short labels are furniture."""
    repeated_sentence = "This is a normal sentence of prose that repeats."
    pages = [f"Title\n{repeated_sentence}\nunique {i}" for i in range(5)]

    repeated = find_repeated_lines(pages)

    assert _normalise(repeated_sentence) not in repeated


def test_short_labels_at_page_edges_are_furniture():
    """A one-word running head repeated on every page is furniture."""
    pages = [f"Intro\nBody of page {i}\nmore text here" for i in range(5)]

    assert "intro" in find_repeated_lines(pages)


def test_long_repeated_lines_are_not_furniture():
    long_line = "A" * 200
    pages = [f"{long_line}\nunique {i}" for i in range(5)]

    assert find_repeated_lines(pages) == set()


def test_lines_too_short_to_carry_meaning_are_not_furniture():
    pages = [f"ab\nBody {i}" for i in range(5)]

    assert find_repeated_lines(pages) == set()


def test_repeated_line_in_body_is_not_furniture():
    """Furniture only lives at page edges, so a body phrase is safe."""
    body_repeat = "Shared body phrase"
    pages = [
        f"Header {i}\nfiller1\nfiller2\n{body_repeat}\nfiller3\nfiller4\nfiller5"
        for i in range(5)
    ]

    assert _normalise(body_repeat) not in find_repeated_lines(pages)


def _normalise(line: str) -> str:
    return " ".join(line.split()).casefold()


def test_remove_repeated_lines_drops_only_matches():
    text = "JWST Handbook\nReal content\nConfidential"

    cleaned = remove_repeated_lines(text, {"jwst handbook", "confidential"})

    assert cleaned == "Real content"


def test_remove_repeated_lines_is_case_insensitive():
    assert remove_repeated_lines("CONFIDENTIAL\nbody", {"confidential"}) == "body"


def test_remove_repeated_lines_with_nothing_to_remove():
    assert remove_repeated_lines("body", set()) == "body"


# --- clean_pages on loader units -------------------------------------------

def test_clean_pages_cleans_each_unit():
    pages = [Page("Body   text\nPage 12\nMore"), Page("Other    text")]

    cleaned = clean_pages(pages)

    assert cleaned[0].text == "Body text\nMore"
    assert cleaned[1].text == "Other text"


def test_clean_pages_preserves_metadata():
    pages = [Page("text", source="a.pdf", page=3)]

    cleaned = clean_pages(pages)

    assert cleaned[0].metadata["source"] == "a.pdf"
    assert cleaned[0].metadata["page"] == 3


def test_clean_pages_updates_char_count():
    pages = [Page("too     many     spaces")]

    cleaned = clean_pages(pages)

    assert cleaned[0].char_count == len("too many spaces")


def test_clean_pages_removes_document_furniture():
    pages = [
        Page(f"JWST Guide\nBody {i}\nPage {i} of 4", page=i) for i in range(1, 5)
    ]

    cleaned = clean_pages(pages)

    for page in cleaned:
        assert "JWST Guide" not in page.text
        assert f"Page {page.metadata['page']}" not in page.text
        assert f"Body {page.metadata['page']}" in page.text


def test_clean_pages_furniture_removal_can_be_disabled():
    pages = [Page(f"JWST Guide\nBody {i}", page=i) for i in range(1, 5)]

    cleaned = clean_pages(pages, remove_furniture=False)

    assert "JWST Guide" in cleaned[0].text


def test_clean_pages_accepts_mappings():
    pages = [{"text": "Body    text", "metadata": {"page": 1}, "char_count": 13}]

    cleaned = clean_pages(pages)

    assert cleaned[0]["text"] == "Body text"
    assert cleaned[0]["char_count"] == len("Body text")


def test_clean_pages_handles_no_pages():
    assert clean_pages([]) == []


def test_clean_pages_does_not_merge_units():
    """Cleaning must never join pages: that would destroy page citations."""
    pages = [Page("Page one text", page=1), Page("Page two text", page=2)]

    cleaned = clean_pages(pages)

    assert len(cleaned) == 2
    assert cleaned[0].metadata["page"] == 1
    assert cleaned[1].metadata["page"] == 2
    assert cleaned[0].text == "Page one text"
    assert cleaned[1].text == "Page two text"


def test_clean_page_handles_a_single_unit():
    page = Page("Body    text\nPage 12")

    cleaned = clean_page(page)

    assert cleaned.text == "Body text"


# --- integration with chunking ---------------------------------------------

def test_cleaned_text_chunks_with_metadata_intact():
    from src.ingestion.chunker import chunk_pages

    pages = [
        Page(f"JWST Guide\nRefunds   within   30 days. Body {i}\nPage {i} of 3",
             source="guide.pdf", page=i)
        for i in range(1, 4)
    ]

    chunks = chunk_pages(clean_pages(pages))

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "guide.pdf"
        assert chunk.metadata["page"] in (1, 2, 3)
        assert "JWST Guide" not in chunk.text
        assert "   " not in chunk.text
