# app.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from psycopg2.extras import Json
import json
import uuid
import os
import streamlit as st
import logging
import psycopg2
from src.config.config import CONFIG
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.generator import Generator
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
def load_pipeline():
    """Load all RAG components and cache them."""
    logger.info("Loading RAG pipeline...")

    # 1. Database connection (PostgreSQL)
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_metadata"),
        user=os.getenv("POSTGRES_USER", "raglab"),
        password=os.getenv("POSTGRES_PASSWORD", "raglab"),
    )
    conn.autocommit = True

    # 2. Embedder & Indexer (Qdrant)
    embedder = Embedder()
    indexer = Indexer()
    indexer.ensure_collection()

    # 3. Base Retrievers
    dense = DenseRetriever(embedder, indexer)
    sparse = SparseRetriever(db_conn=conn, config=CONFIG)
    hybrid = HybridRetriever(dense, sparse)

    # 4. Advanced Retrievers
    # LLM instance (Ollama)
    llm = ChatOllama(
        model=CONFIG["hyde"].get("model", "llama3.1:8b"),
        temperature=CONFIG["hyde"].get("temperature", 0.0),
    )

    # HyDE
    hyde = HyDERetriever(
        llm=llm,
        embedder=embedder,
        base_retriever=hybrid,
        config=CONFIG,
    )

    # Multi-Query
    multi_query = MultiQueryRetriever(
        llm=llm,
        base_retriever=hybrid,
        config=CONFIG,
    )

    # Reranked
    reranked = CrossEncoderReranker(
        base_retriever=hybrid,
        config=CONFIG,
    )

    # Generator (with a default retriever - hybrid)
    generator = Generator(retriever=hybrid)

    logger.info("RAG pipeline loaded successfully.")
    return {
        "generator": generator,
        "retrievers": {
            "Hybrid": hybrid,
            "HyDE": hyde,
            "Multi-Query": multi_query,
            "Reranked": reranked,
        }
    }