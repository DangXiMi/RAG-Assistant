# src/ingestion/cleaning.py
"""Text normalisation applied between loading and chunking.

Extracted text carries artefacts that hurt retrieval: hard-wrapped lines,
hyphenated line breaks, repeated page furniture (headers, footers, page
numbers), non-breaking spaces and stray control characters. Cleaning is
deterministic and dependency-free so results are reproducible.

Ordering matters. Cleaning runs on each loader unit *after* page/sheet
boundaries exist and *before* chunking, so it can never merge pages back
together and destroy the citation metadata built in the loaders.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# Control characters except tab (09) and newline (10). Form feeds and vertical
# tabs (0x0b, 0x0c) are common PDF page separators and must go too.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# A line ending in a hyphen that continues on the next line: "informa-\ntion"
# becomes "information". Only joins when the next line starts lowercase, which
# avoids breaking legitimately hyphenated phrases like "State-\nof-the-Art".
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n[ \t]*([a-z])")

# A line that is explicitly labelled as pagination: "Page 12", "12 of 40",
# "Page 12 of 40". A bare number is NOT matched here, because without document
# context a lone "1990" is content, not a page number.
_EXPLICIT_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:[-–—\[\(]*\s*)?"
    r"(?:page\s+\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
    r"|\d{1,4}\s*(?:of|/)\s*\d{1,4})"
    r"\s*(?:[-–—\]\)]*)\s*$",
    re.IGNORECASE,
)

# A bare number with no "Page" prefix. Stripped only when the document proves
# the number is pagination (it counts up across pages); otherwise a standalone
# number is content, e.g. a year heading.
_BARE_NUMBER_LINE = re.compile(r"^\s*[-–—\[\(]*\s*(\d{1,4})\s*[-–—\]\)]*\s*$")

# Runs of 3+ newlines and 2+ spaces, collapsed after the line-level fixes.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")

# Characters that are whitespace-like but not the ASCII space, normalised to it
# so downstream splitting and FTS behave consistently.
_ODD_SPACES = {
    "\u00a0": " ",  # no-break space
    "\u2007": " ",  # figure space
    "\u202f": " ",  # narrow no-break space
    "\u2009": " ",  # thin space
    "\u200b": "",   # zero-width space
    "\ufeff": "",   # BOM / zero-width no-break space
    "\u00ad": "",   # soft hyphen
}

# A line repeated across pages is furniture only if it is short. Long repeated
# lines are more likely to be real content (a standard disclaimer, a table).
_MAX_FURNITURE_CHARS = 120
_MAX_FURNITURE_WORDS = 15

# Fraction of pages a line must appear on before it counts as furniture.
_FURNITURE_PAGE_RATIO = 0.6

# Never treat a line this short as furniture: a shared one-word heading such as
# "Introduction" carries meaning.
_MIN_FURNITURE_CHARS = 4

# How close to the top/bottom edge of a page a line must be to be furniture.
_EDGE_LINES = 3


def _strip_line_noise(text: str, *, strip_page_numbers: bool = True) -> str:
    """Apply the per-line rules, dropping explicitly labelled page numbers.

    Bare numeric lines are deliberately kept here; see `_bare_pagination_keys`
    for the document-level check that removes them when they really are
    pagination.
    """
    if not strip_page_numbers:
        return text

    kept = []
    for line in text.split("\n"):
        if _EXPLICIT_PAGE_NUMBER_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _bare_pagination_keys(texts: Sequence[str]) -> set[str]:
    """Find standalone numbers that are genuinely pagination.

    A bare number is only furniture when the document proves it: the numbers
    form an ascending sequence across consecutive pages (1, 2, 3...). A lone
    "1990" on every page is content, such as a year heading, and must survive.
    """
    numbers: list[int | None] = []
    keys: list[str | None] = []

    for text in texts:
        value, key = None, None
        for line in text.split("\n"):
            match = _BARE_NUMBER_LINE.match(line)
            if match:
                value = int(match.group(1))
                key = _normalise_for_comparison(line)
                break
        numbers.append(value)
        keys.append(key)

    if len(numbers) < 3 or any(n is None for n in numbers):
        return set()

    # Ascending with a step of one across the whole document.
    if all(
        numbers[i + 1] - numbers[i] == 1  # type: ignore[operator]
        for i in range(len(numbers) - 1)
    ):
        return {k for k in keys if k is not None}

    return set()


def clean_text(text: str, *, fix_hyphenation: bool = True,
               strip_page_numbers: bool = True) -> str:
    """Normalise a single block of extracted text.

    Args:
        text: Raw text from a loader.
        fix_hyphenation: Rejoin words broken across lines by a hyphen.
        strip_page_numbers: Drop lines that are only a page number.

    Returns:
        Cleaned text with consistent whitespace. Never returns None.
    """
    if not text:
        return ""

    # 1. Unicode normalisation, then collapse odd space characters.
    value = unicodedata.normalize("NFKC", text)
    for source, replacement in _ODD_SPACES.items():
        value = value.replace(source, replacement)

    # 2. Remove control characters that would corrupt chunks or FTS.
    value = _CONTROL_CHARS.sub("", value)

    # 3. Normalise line endings before line-oriented rules run.
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Rejoin words split across a line break.
    if fix_hyphenation:
        value = _HYPHEN_LINEBREAK.sub(r"\1\2", value)

    # 5. Strip per-line noise.
    value = _strip_line_noise(value, strip_page_numbers=strip_page_numbers)

    # 6. Collapse runs of spaces and blank lines.
    value = _EXCESS_SPACES.sub(" ", value)
    value = _EXCESS_BLANK_LINES.sub("\n\n", value)

    return value.strip()


def _normalise_for_comparison(line: str) -> str:
    """Key used to detect repeated lines, ignoring case and spacing."""
    return " ".join(line.split()).casefold()


def _is_furniture_candidate(line: str) -> bool:
    """Whether a line is short enough to plausibly be a header or footer."""
    stripped = line.strip()
    if len(stripped) < _MIN_FURNITURE_CHARS or len(stripped) > _MAX_FURNITURE_CHARS:
        return False
    if len(stripped.split()) > _MAX_FURNITURE_WORDS:
        return False
    # A line ending in sentence punctuation is prose, not furniture.
    if stripped[-1] in ".:;,!?":
        return False
    return True


def find_repeated_lines(pages: Sequence[str]) -> set[str]:
    """Identify header/footer lines repeated across the majority of pages.

    Considers only lines near the top or bottom edge of a page, so a phrase that
    happens to recur in body text is not mistaken for page furniture.

    Args:
        pages: Per-page text, in order.

    Returns:
        Set of comparison keys (see `_normalise_for_comparison`) to drop.
    """
    if len(pages) < 3:
        # With fewer than three pages there is no reliable pattern to detect,
        # and dropping a line could remove genuine content.
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        lines = page.split("\n")
        if len(lines) > _EDGE_LINES * 2:
            edge_lines = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
        else:
            edge_lines = lines

        seen_on_this_page = set()
        for line in edge_lines:
            if not _is_furniture_candidate(line):
                continue
            key = _normalise_for_comparison(line)
            if key in seen_on_this_page:
                continue
            seen_on_this_page.add(key)
            counts[key] = counts.get(key, 0) + 1

    threshold = max(3, int(len(pages) * _FURNITURE_PAGE_RATIO))
    return {key for key, count in counts.items() if count >= threshold}


def remove_repeated_lines(text: str, repeated: Iterable[str]) -> str:
    """Drop lines whose comparison key is in `repeated`."""
    repeated = set(repeated)
    if not repeated or not text:
        return text

    kept = [
        line
        for line in text.split("\n")
        if _normalise_for_comparison(line) not in repeated
    ]
    return "\n".join(kept)


def clean_page(
    page: Any,
    *,
    fix_hyphenation: bool = True,
    strip_page_numbers: bool = True,
) -> Any:
    """Clean one loader unit in place, preserving its metadata.

    Accepts a `LoadedPage`-like object or a mapping, and returns the same shape.
    """
    if isinstance(page, dict):
        text = page.get("text") or ""
        cleaned = clean_text(
            text,
            fix_hyphenation=fix_hyphenation,
            strip_page_numbers=strip_page_numbers,
        )
        page["text"] = cleaned
        if "char_count" in page:
            page["char_count"] = len(cleaned)
        return page

    cleaned = clean_text(
        getattr(page, "text", "") or "",
        fix_hyphenation=fix_hyphenation,
        strip_page_numbers=strip_page_numbers,
    )

    page.text = cleaned
    # Keep the cached length consistent; is_empty and logging depend on it.
    try:
        page.char_count = len(cleaned)
    except AttributeError:  # pragma: no cover - defensive for odd objects
        pass
    return page


def clean_pages(
    pages: Sequence[Any],
    *,
    remove_furniture: bool = True,
    fix_hyphenation: bool = True,
    strip_page_numbers: bool = True,
) -> list[Any]:
    """Clean every unit of a document, then strip repeated page furniture.

    Furniture detection needs the whole document to know what repeats, so this
    takes the full unit list rather than one page at a time.

    Args:
        pages: Loader output, in order.
        remove_furniture: Detect and drop repeated headers/footers.
        fix_hyphenation: Rejoin words split across line breaks.
        strip_page_numbers: Drop lines that are only a page number.

    Returns:
        A new list of cleaned units (the originals are mutated in place too,
        since callers hold references to them).
    """
    pages = list(pages)
    if not pages:
        return []

    texts = [
        clean_text(
            (p.get("text") if isinstance(p, dict) else getattr(p, "text", "")) or "",
            fix_hyphenation=fix_hyphenation,
            strip_page_numbers=strip_page_numbers,
        )
        for p in pages
    ]

    repeated: set[str] = set()
    if remove_furniture:
        repeated = find_repeated_lines(texts)
        if repeated:
            # Log the keys, not the raw lines, to keep the message short.
            sample = sorted(repeated)[:3]
            logger.info(
                "Removed %d repeated header/footer line pattern(s), e.g. %s",
                len(repeated),
                sample,
            )

    # Bare numbers are only removed when the document proves they are
    # pagination; `clean_text` alone cannot know that.
    if strip_page_numbers:
        repeated |= _bare_pagination_keys(texts)

    cleaned_pages = []
    for page, text in zip(pages, texts):
        if repeated:
            text = remove_repeated_lines(text, repeated)
            text = _EXCESS_BLANK_LINES.sub("\n\n", text).strip()

        if isinstance(page, dict):
            page["text"] = text
            if "char_count" in page:
                page["char_count"] = len(text)
            cleaned_pages.append(page)
        else:
            page.text = text
            try:
                page.char_count = len(text)
            except AttributeError:  # pragma: no cover
                pass
            cleaned_pages.append(page)

    return cleaned_pages
