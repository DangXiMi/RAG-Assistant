# scripts/run_evaluation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import psycopg2
import os
from src.config.config import CONFIG
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.generator import Generator
from src.evaluation.ragas_evaluator import RAGASEvaluator
from langchain_ollama import ChatOllama
from scripts.test_e2e import SAMPLE_DOCS, seed_postgres, seed_qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline():
    """Load all RAG components."""
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
    seed_postgres(conn)

    # 2. Embedder & Indexer (Qdrant)
    embedder = Embedder()
    indexer = Indexer()
    logger.info(f"Qdrant collection: {indexer.collection_name}")
    seed_qdrant(indexer, embedder)
    
    # 3. Base Retrievers
    dense = DenseRetriever(embedder, indexer)
    sparse = SparseRetriever(db_conn=conn, config=CONFIG)
    hybrid = HybridRetriever(dense, sparse)

    # 4. Advanced Retrievers
    llm = ChatOllama(
        model=CONFIG["hyde"].get("model", "llama3.1:8b"),
        temperature=CONFIG["hyde"].get("temperature", 0.0),
    )

    hyde = HyDERetriever(
        llm=llm,
        embedder=embedder,
        base_retriever=hybrid,
        config=CONFIG,
    )

    multi_query = MultiQueryRetriever(
        llm=llm,
        base_retriever=hybrid,
        config=CONFIG,
    )

    reranked = CrossEncoderReranker(
        base_retriever=hybrid,
        config=CONFIG,
    )

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


def main():
    logger.info("Loading pipeline...")
    pipeline = load_pipeline()
    generator = pipeline["generator"]
    retrievers = pipeline["retrievers"]

    evaluator = RAGASEvaluator(generator=generator)
    results = {}

    # Evaluate ALL modes
    for mode_name, retriever in retrievers.items():
        logger.info(f"Evaluating mode: {mode_name}")
        try:
            metrics = evaluator.evaluate(retriever, top_k=3)
            results[mode_name] = metrics
            logger.info(f"✅ {mode_name} evaluation complete.")
            logger.info(f"{mode_name} \n {metrics} .")
        except Exception as e:
            logger.error(f"❌ {mode_name} evaluation failed: {e}")

    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Aggregated)")
    print("="*60)
    summary = {}
    for mode, metrics in results.items():
        print(f"\n{mode}:")
        for k, v in metrics["aggregated"].items():
            print(f"  {k}: {v:.3f}")
        summary[mode] = metrics["aggregated"]

    # Save to CSV for dashboard
    if summary:
        df = pd.DataFrame(summary).T
        # Rename index column to "Mode"
        df.index.name = "Mode"
        df.to_csv("data/evaluation/metrics.csv")
        logger.info("Metrics saved to data/evaluation/metrics.csv")
    else:
        logger.warning("No metrics were collected. Check your evaluator.")


if __name__ == "__main__":
    main()