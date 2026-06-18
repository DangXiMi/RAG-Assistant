# src/ingestion/chunker.py
import uuid
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import CHUNK_SIZE, OVERLAP, SEPARATORS


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