# test_chunking.py
import uuid
import pytest
from src.ingestion.chunker import CHUNK_SIZE, OVERLAP
from src.ingestion.chunker import chunk_text



def test_short_text_returns_one_chunk():
    text = "This is a short document."
    metadata = {"source": "test"}

    chunks = chunk_text(text, metadata)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.text == text
    assert chunk.start_char == 0
    assert chunk.end_char == len(text)

    assert chunk.metadata["source"] == "test"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["total_chunks"] == 1


def test_empty_string():
    chunks = chunk_text("", {"source": "empty"})

    # Adjust depending on your implementation choice
    assert len(chunks) in (0, 1)

    if len(chunks) == 1:
        assert chunks[0].text == ""


def test_exact_chunk_size_boundary():
    text = "A" * CHUNK_SIZE

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_one_character_over_limit():
    text = "A" * (CHUNK_SIZE + 1)

    chunks = chunk_text(text)

    assert len(chunks) >= 2

    assert len(chunks[0].text) <= CHUNK_SIZE
    assert len(chunks[-1].text) > 0


def test_long_paragraph_splits_into_multiple_chunks():
    sentence = "This is a test sentence. "
    text = sentence * 300

    chunks = chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) > 0

def test_metadata_propagation():
    metadata = {
        "source": "manual",
        "document_id": "doc-123",
    }

    text = "test " * 3000

    chunks = chunk_text(text, metadata)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "manual"
        assert chunk.metadata["document_id"] == "doc-123"

        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_uuid_generation():
    text = "test " * 3000

    chunks = chunk_text(text)

    ids = []

    for chunk in chunks:
        uuid.UUID(str(chunk.id))
        ids.append(str(chunk.id))

    assert len(ids) == len(set(ids))


def test_start_end_positions_are_valid():
    text = ("ABC DEF GHI. " * 500)

    chunks = chunk_text(text)

    for chunk in chunks:
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.end_char <= len(text)


def test_start_end_positions_are_monotonic():
    text = ("Lorem ipsum dolor sit amet. " * 500)

    chunks = chunk_text(text)

    assert chunks[0].start_char == 0

    for i in range(len(chunks) - 1):
        assert chunks[i].start_char <= chunks[i + 1].start_char
        assert chunks[i].end_char <= chunks[i + 1].end_char


def test_chunk_indexes_are_sequential():
    text = "test " * 3000

    chunks = chunk_text(text)

    indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
    ]

    assert indexes == list(range(len(chunks)))


def test_total_chunks_consistency():
    text = "test " * 3000

    chunks = chunk_text(text)

    total = len(chunks)

    for chunk in chunks:
        assert chunk.metadata["total_chunks"] == total


def test_sentence_aware_splitting():
    text = (
        "Sentence one. "
        "Sentence two. "
        "Sentence three. "
    ) * 200

    chunks = chunk_text(text)

    assert len(chunks) > 1

    # Most RecursiveCharacterTextSplitter implementations
    # preserve sentence boundaries when possible.
    for chunk in chunks[:-1]:
        assert chunk.text.strip()


def test_paragraph_aware_splitting():
    text = (
        "Paragraph 1.\n\n"
        + ("A" * 1800)
        + "\n\n"
        + ("B" * 1800)
        + "\n\n"
        + ("C" * 1800)
    )

    chunks = chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) > 0


def test_word_level_fallback():
    text = ("word " * 5000)

    chunks = chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) > 0


def test_character_level_fallback():
    text = "A" * 10000

    chunks = chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_SIZE + OVERLAP


def test_unicode_text():
    text = (
        "Xin chào thế giới. "
        "こんにちは世界。 "
        "안녕하세요 세계. "
        "مرحبا بالعالم. "
    ) * 300

    chunks = chunk_text(text)

    assert len(chunks) > 0

    reconstructed = "".join(
        chunk.text
        if i == 0
        else chunk.text[OVERLAP:]
        for i, chunk in enumerate(chunks)
    )

    assert len(reconstructed) > 0


def test_newline_separator_priority():
    text = (
        ("A" * 500)
        + "\n"
        + ("B" * 500)
        + "\n"
        + ("C" * 500)
        + "\n"
        + ("D" * 500)
        + "\n"
        + ("E" * 500)
    )

    chunks = chunk_text(text)

    assert len(chunks) >= 2


def test_chunk_has_required_fields():
    text = "test " * 3000

    chunks = chunk_text(text)

    for chunk in chunks:
        assert hasattr(chunk, "id")
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "metadata")
        assert hasattr(chunk, "start_char")
        assert hasattr(chunk, "end_char")



def test_grounding_offsets_match_original_text():
    text = ("Grounding test sentence. " * 500)

    chunks = chunk_text(text)

    for chunk in chunks:
        extracted = text[
            chunk.start_char:chunk.end_char
        ]

        # If offsets represent the actual chunk span,
        # this should be exact.
        assert extracted == chunk.text