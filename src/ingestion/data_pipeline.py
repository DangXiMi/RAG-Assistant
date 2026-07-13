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


def seed_postgres(conn, docs):
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

    for text in docs:
        doc_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO chunks (id, text, metadata) VALUES (%s, %s, %s)",
            (doc_id, text, Json({"source": "test"}))
        )
    conn.commit()
    logger.info(f"Inserted {len(docs)} documents into PostgreSQL")


def seed_qdrant(indexer, embedder, chunked, vectors):
    """Chunk, embed, and index documents into Qdrant."""
    
    logger.info(f"Created {len(chunked)} chunks from sample documents")
    
    # Embed all chunks
    texts = [c.text for c in chunked]
    vectors = embedder.embed(texts)
    
    # Index into Qdrant
    indexer.ensure_collection()
    indexer.index(chunked, vectors)
    logger.info(f"Indexed {len(chunked)} chunks into Qdrant")
    
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