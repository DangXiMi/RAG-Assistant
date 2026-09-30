# src/ingestion/chunker.py
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import yaml
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import CHUNK_SIZE, OVERLAP, SEPARATORS

logger = logging.getLogger(__name__)

# Metadata keys carried by loader output that must always survive chunking,
# because citations depend on them.
_CITATION_KEYS = ("source", "page", "page_count", "sheet", "location", "row")


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: dict
    start_char: int
    end_char: int


def _compute_offsets(original_text: str, chunks: List[str], overlap: int) -> List[tuple]:
    """
    Computes start/end character offsets for each chunk.
    We walk through the original text, matching each chunk sequentially.
    """
    offsets = []
    cursor = 0
    for chunk_text in chunks:
        start = original_text.find(chunk_text, cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk_text)
        offsets.append((start, end))
        cursor = end - overlap if end - overlap > start else end
    return offsets


def chunk_text(text: str, metadata: Optional[dict] = None) -> List[Chunk]:
    if metadata is None:
        metadata = {}

    if not text:
        return [Chunk(
            text="",
            metadata={**metadata, "chunk_index": 0, "total_chunks": 1},
            start_char=0,
            end_char=0
        )]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len,
        separators=SEPARATORS,  
        keep_separator=False,
    )

    raw_chunks = splitter.split_text(text)
    offsets = _compute_offsets(text, raw_chunks, OVERLAP)
    total = len(raw_chunks)

    result = []
    for i, (chunk_text, (start, end)) in enumerate(zip(raw_chunks, offsets)):
        new_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": total,
        }
        result.append(Chunk(
            text=chunk_text,
            metadata=new_meta,
            start_char=start,
            end_char=end,
        ))
    return result


def _page_text_and_metadata(page: Any) -> tuple[str, Dict[str, Any]]:
    """Accept either a `LoadedPage` or a plain `(text, metadata)` mapping.

    Keeping this duck-typed means the chunker does not import the loaders
    package, so the two stay independently testable.
    """
    if isinstance(page, dict):
        text = page.get("text", "") or ""
        metadata = dict(page.get("metadata", {}) or {})
        return text, metadata

    text = getattr(page, "text", "") or ""
    metadata = dict(getattr(page, "metadata", {}) or {})
    return text, metadata


def chunk_pages(
    pages: Sequence[Any],
    metadata: Optional[dict] = None,
    *,
    skip_empty: bool = True,
) -> List[Chunk]:
    """Chunk a document page by page, preserving per-page citation metadata.

    Each page (or sheet) is split independently, so every resulting chunk
    carries the `page` / `sheet` of the unit it came from and can be cited
    precisely. `chunk_index` and `total_chunks` are global across the whole
    document, so ordering is unchanged from the single-string flow.

    Args:
        pages: Ordered loader output -- `LoadedPage` objects or mappings with
            `text` and `metadata` keys.
        metadata: Optional document-level metadata merged into every chunk.
            Per-page metadata takes precedence on key collisions, otherwise a
            document-level `page` would overwrite the real page number.
        skip_empty: Drop pages that contain no text. An empty page still
            consumes a page number in the source document, so its metadata is
            simply unused rather than misleading.

    Returns:
        A flat, document-ordered list of `Chunk`.
    """
    document_metadata = dict(metadata or {})

    # Split each page first so we know the true total before assigning indexes.
    per_page: List[tuple[str, Dict[str, Any], List[str], List[tuple]]] = []
    total = 0

    for page in pages:
        text, page_metadata = _page_text_and_metadata(page)

        if skip_empty and not text.strip():
            continue

        combined = {**document_metadata, **page_metadata}
        # Never let a document-level value shadow the real location.
        for key in _CITATION_KEYS:
            if key in page_metadata:
                combined[key] = page_metadata[key]

        split = chunk_text(text, combined)
        raw = [c.text for c in split]
        offsets = [(c.start_char, c.end_char) for c in split]

        per_page.append((text, combined, raw, offsets))
        total += len(split)

    if total == 0:
        logger.warning("chunk_pages produced no chunks: every page was empty")
        return []

    result: List[Chunk] = []
    index = 0

    for _text, combined, raw, offsets in per_page:
        for chunk_body, (start, end) in zip(raw, offsets):
            result.append(
                Chunk(
                    text=chunk_body,
                    metadata={
                        **combined,
                        "chunk_index": index,
                        "total_chunks": total,
                    },
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

    return result