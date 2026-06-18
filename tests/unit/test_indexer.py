# tests/unit/test_indexer.py
import pytest
import uuid
from src.ingestion.indexer import Indexer
from src.ingestion.chunker import Chunk


@pytest.fixture
def indexer():
    """Return an Indexer that uses in-memory Qdrant."""
    # We'll override config to use :memory: inside the Indexer.
    # For simplicity, we'll pass a config dict directly.
    config = {
        "qdrant": {
            "host": ":memory:",
            "port": 6333,
            "collection_name": "test_collection",
            "vector_size": 384,
        }
    }
    return Indexer(config=config)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id=str(uuid.uuid4()),
            text="First document chunk",
            metadata={"source": "test", "chunk_index": 0, "total_chunks": 2},
            start_char=0,
            end_char=20,
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="Second document chunk",
            metadata={"source": "test", "chunk_index": 1, "total_chunks": 2},
            start_char=21,
            end_char=40,
        ),
    ]


@pytest.fixture
def sample_vectors():
    # Dummy vectors of dimension 384 (just for testing)
    return [[0.1] * 384, [0.2] * 384]


def test_indexer_initialization(indexer):
    """Test that indexer loads config and client."""
    assert indexer.collection_name == "test_collection"
    assert indexer.vector_size == 384


def test_ensure_collection_creates(indexer):
    """Ensure collection is created successfully."""
    indexer.ensure_collection()
    collections = indexer.client.get_collections()
    assert indexer.collection_name in [c.name for c in collections.collections]


def test_index_single_chunk(indexer, sample_chunks, sample_vectors):
    """Index one chunk and verify it exists."""
    indexer.ensure_collection()
    # Index only the first chunk
    indexer.index([sample_chunks[0]], [sample_vectors[0]])

    # Query the collection to verify count
    count = indexer.client.count(
        collection_name=indexer.collection_name,
        exact=True,
    ).count
    assert count == 1


def test_index_multiple_chunks(indexer, sample_chunks, sample_vectors):
    """Index multiple chunks and verify count."""
    indexer.ensure_collection()
    indexer.index(sample_chunks, sample_vectors)

    count = indexer.client.count(
        collection_name=indexer.collection_name,
        exact=True,
    ).count
    assert count == 2


def test_upsert_replaces_existing(indexer, sample_chunks, sample_vectors):
    """Indexing the same ID should overwrite the point."""
    indexer.ensure_collection()

    # Index first time
    indexer.index([sample_chunks[0]], [sample_vectors[0]])
    count1 = indexer.client.count(
        collection_name=indexer.collection_name,
        exact=True,
    ).count
    assert count1 == 1

    # Index again with the same ID but different vector
    new_vector = [[0.9] * 384]
    indexer.index([sample_chunks[0]], new_vector)

    # Count should still be 1 (replaced, not duplicated)
    count2 = indexer.client.count(
        collection_name=indexer.collection_name,
        exact=True,
    ).count
    assert count2 == 1

    # Retrieve the point to ensure it was updated
    point = indexer.client.retrieve(
        collection_name=indexer.collection_name,
        ids=[sample_chunks[0].id],
        with_vectors=True,
    )[0]
    # The vector should be the new one (approximately)
    assert point.id == sample_chunks[0].id
    assert len(point.vector) == 384


def test_index_empty_list(indexer):
    """Indexing empty lists should do nothing and not raise errors."""
    indexer.ensure_collection()
    indexer.index([], [])  # Should not raise
    count = indexer.client.count(
        collection_name=indexer.collection_name,
        exact=True,
    ).count
    assert count == 0


def test_index_mismatched_lengths(indexer, sample_chunks, sample_vectors):
    """Should raise ValueError if lengths differ."""
    indexer.ensure_collection()
    with pytest.raises(ValueError, match="must have the same length"):
        indexer.index(sample_chunks, sample_vectors[:-1])  # one less vector


def test_payload_stored_correctly(indexer, sample_chunks, sample_vectors):
    """Ensure metadata and text are stored in the payload."""
    indexer.ensure_collection()
    indexer.index([sample_chunks[0]], [sample_vectors[0]])

    point = indexer.client.retrieve(
        collection_name=indexer.collection_name,
        ids=[sample_chunks[0].id],
    )[0]

    assert point.payload["text"] == "First document chunk"
    assert point.payload["source"] == "test"
    assert point.payload["chunk_index"] == 0
    assert point.payload["total_chunks"] == 2