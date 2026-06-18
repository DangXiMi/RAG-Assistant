# tests/unit/test_embedder.py
import pytest
from src.ingestion.embedder import Embedder
from src.ingestion.chunker import CHUNK_SIZE, OVERLAP 


@pytest.fixture(scope="session")
def embedder():
    """Load the embedder once per test session."""
    # We'll load config inside the embedder, but for tests we can
    # rely on the default config/default.yaml
    return Embedder()


def test_embedder_initialization(embedder):
    """Test that the embedder loads the model and reports dimensions."""
    assert embedder.dimension == 384  # all-MiniLM-L6-v2 is 384
    assert embedder.device in ["cpu", "cuda"]


def test_embed_single_text(embedder):
    """Embedding a single string returns a list of length 1."""
    texts = ["Hello world"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimension
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_multiple_texts(embedder):
    """Embedding multiple strings returns equal number of vectors."""
    texts = ["Hello", "World", "This is a test", "Sentence four"]
    vectors = embedder.embed(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == embedder.dimension


def test_embed_batch_size_respect(embedder):
    """Ensure batching doesn't change output shape."""
    # Create 100 texts
    texts = [f"Document number {i}" for i in range(100)]
    vectors = embedder.embed(texts)
    assert len(vectors) == 100
    # Spot check first and last
    assert len(vectors[0]) == embedder.dimension
    assert len(vectors[99]) == embedder.dimension


def test_embed_empty_list(embedder):
    """Return empty list for empty input."""
    vectors = embedder.embed([])
    assert vectors == []


def test_embed_deterministic(embedder):
    """Embedding the same text twice yields identical vectors."""
    text = "The quick brown fox jumps over the lazy dog"
    vec1 = embedder.embed([text])[0]
    vec2 = embedder.embed([text])[0]
    assert vec1 == vec2  # exact floating point match


def test_embed_unicode_texts(embedder):
    """Handle non-ASCII text gracefully."""
    texts = [
        "Xin chào thế giới",
        "こんにちは世界",
        "안녕하세요 세계",
        "مرحبا بالعالم"
    ]
    vectors = embedder.embed(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == embedder.dimension


def test_embed_empty_string(embedder):
    """Embedding an empty string should still produce a vector (not crash)."""
    texts = [""]
    vectors = embedder.embed(texts)
    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimension


def test_embed_mixed_empty_and_normal(embedder):
    """Mix of empty and normal strings."""
    texts = ["", "Normal text", ""]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == embedder.dimension
    # Check that empty vector is not all zeros (or if it is, that's fine)
    # We just care about shape.


def test_embedder_config_read():
    """Ensure embedder reads batch_size from config correctly."""
    # We can check that the embedder actually uses the config.
    # Since we don't expose batch_size, we can just instantiate
    # and ensure no error.
    embedder = Embedder()
    assert hasattr(embedder, "batch_size")
    assert embedder.batch_size > 0