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


def main():
    logger.info("=== Starting E2E Smoke Test ===")

    # Seed both stores through the real ingestion components. The old local
    # seeders ran DROP TABLE and minted throwaway ids, which emptied the sparse
    # index of everything else and stopped RRF from fusing the two stores.
    # `replace_source` only refreshes this corpus's own rows.
    from scripts.seed_corpus_lib import DEFAULT_COLLECTION, replace_source

    replace_source(SAMPLE_DOCS, source_name="e2e_test.txt", collection_name=DEFAULT_COLLECTION)

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_metadata"),
        user=os.getenv("POSTGRES_USER", "raglab"),
        password=os.getenv("POSTGRES_PASSWORD", "raglab"),
    )
    conn.autocommit = True

    embedder = Embedder()

    from src.config.config import CONFIG as _CONFIG

    indexer = Indexer(
        config={
            **_CONFIG,
            "qdrant": {**_CONFIG["qdrant"], "collection_name": DEFAULT_COLLECTION},
        }
    )
    logger.info(f"Qdrant collection: {indexer.collection_name}")

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
        
    print("=" * 50)
    print("DENSE")
    dense_docs = dense.search(test_query, top_k=3)
    print("count:", len(dense_docs))
    for d in dense_docs:
        print(d)

    print("=" * 50)
    print("SPARSE")
    sparse_docs = sparse.search(test_query, top_k=3)
    print("count:", len(sparse_docs))
    for d in sparse_docs:
        print(d)

    print("=" * 50)
    print("HYBRID")
    hybrid_docs = hybrid.search(test_query, top_k=3)
    print("count:", len(hybrid_docs))
    for d in hybrid_docs:
        print(d)

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