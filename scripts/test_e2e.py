# scripts/test_e2e.py
import os
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import Json

from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import Generator
from src.config.config import CONFIG

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 1. SAMPLE DOCUMENTS (Domain: Space exploration)
SAMPLE_DOCS = [
    "The James Webb Space Telescope (JWST) was launched on December 25, 2021. It is the largest optical telescope in space.",
    "JWST has a 6.5-meter diameter mirror, compared to Hubble's 2.4-meter mirror. It operates primarily in the infrared spectrum.",
    "The Hubble Space Telescope was launched in 1990. It has made over 1.5 million observations during its lifetime.",
    "Hubble's images have been used in over 21,000 peer-reviewed scientific papers.",
    "The JWST is positioned at the Sun-Earth L2 Lagrange point, approximately 1.5 million kilometers from Earth.",
    "Hubble orbits Earth at an altitude of about 540 kilometers.",
    "The primary scientific goals of JWST include studying the formation of stars and galaxies, and characterizing exoplanet atmospheres.",
    "Hubble's successor, JWST, is expected to fundamentally change our understanding of the early universe.",
]


def seed_postgres(conn):
    """Insert documents into PostgreSQL with FTS."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    cur.execute("""
        CREATE TABLE chunks (
            id UUID PRIMARY KEY,
            text TEXT NOT NULL,
            metadata JSONB,
            tsv TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(text, '')), 'A')
            ) STORED
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);")
    logger.info("Created chunks table in PostgreSQL")

    for text in SAMPLE_DOCS:
        doc_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO chunks (id, text, metadata) VALUES (%s, %s, %s)",
            (doc_id, text, Json({"source": "e2e_test"}))
        )
    conn.commit()
    logger.info(f"Inserted {len(SAMPLE_DOCS)} documents into PostgreSQL")


def seed_qdrant(indexer, embedder):
    """Chunk, embed, and index documents into Qdrant."""
    chunks = []
    vectors = []
    for text in SAMPLE_DOCS:
        chunked = chunk_text(text, metadata={"source": "e2e_test"})
        chunks.extend(chunked)
    
    logger.info(f"Created {len(chunks)} chunks from sample documents")
    
    # Embed all chunks
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)
    
    # Index into Qdrant
    indexer.ensure_collection()
    indexer.index(chunks, vectors)
    logger.info(f"Indexed {len(chunks)} chunks into Qdrant")


def main():
    logger.info("=== Starting E2E Smoke Test ===")

    # 1. Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_metadata"),
        user=os.getenv("POSTGRES_USER", "raglab"),
        password=os.getenv("POSTGRES_PASSWORD", "raglab"),
    )
    conn.autocommit = True

    # 2. Seed PostgreSQL
    seed_postgres(conn)

    # 3. Initialize Embedder
    embedder = Embedder()
    #logger.info(f"Embedder loaded: dimension={embedder.dimension}")

    # 4. Initialize Indexer (Qdrant)
    indexer = Indexer()
    logger.info(f"Qdrant collection: {indexer.collection_name}")

    # 5. Seed Qdrant
    seed_qdrant(indexer, embedder)

    # 6. Initialize Retrievers
    dense = DenseRetriever(embedder, indexer)
    sparse = SparseRetriever(db_conn=conn, config=CONFIG)
    hybrid = HybridRetriever(dense, sparse)

    # 7. Initialize Generator
    generator = Generator(hybrid)

    # 8. Run a test query
    test_query = "What is the JWST launch date and where is it located?"
    logger.info(f"Test query: '{test_query}'")

    # 9. Manually inspect retrieval results (before generation)
    logger.info("--- Retrieval Results (Hybrid) ---")
    retrieved_docs = hybrid.search(test_query, top_k=3)
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Rank {i+1}: [id={doc['id']}] score={doc['score']:.4f}")
        logger.info(f"  Text: {doc['text'][:150]}...")

    # 10. Generate answer
    logger.info("--- Generating Answer ---")
    result = generator.run(test_query, top_k=3)

    logger.info("=== E2E Test Results ===")
    logger.info(f"ANSWER: {result['answer']}")
    logger.info(f"SOURCES: {result['sources']}")

    # 11. Validate that the answer contains the expected information
    if "December 25, 2021" in result["answer"] and "L2" in result["answer"]:
        logger.info("✅ E2E test PASSED: Answer contains correct facts.")
    else:
        logger.warning("⚠️ E2E test CHECK: Answer may be missing key facts. Check LLM response and retrieval.")

    conn.close()


if __name__ == "__main__":
    main()