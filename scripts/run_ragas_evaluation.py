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
from scripts.test_e2e import SAMPLE_DOCS
from scripts.seed_corpus_lib import DEFAULT_COLLECTION, replace_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline():
    """Load all RAG components against the evaluation corpus."""
    logger.info("Loading RAG pipeline...")

    # Seed both stores through the real ingestion components so chunk ids match
    # between them and RRF can fuse. Only this corpus's own rows are replaced,
    # and it writes to a dedicated collection, so an evaluation run cannot
    # disturb the demo corpus.
    replace_source(SAMPLE_DOCS, source_name="evaluation_corpus.txt")

    # Database connection for the sparse retriever.
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
    indexer = Indexer(
        config={
            **CONFIG,
            "qdrant": {**CONFIG["qdrant"], "collection_name": DEFAULT_COLLECTION},
        }
    )
    logger.info(f"Qdrant collection: {indexer.collection_name}")

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
            logger.info(f"{mode_name} \n {metrics['aggregated']} .")
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

    # Persist per-question scores for every mode. An aggregate alone cannot show
    # whether a 0.8 came from uniformly decent answers or from a few perfect
    # answers hiding several failures.
    per_sample_frames = []
    for mode, metrics in results.items():
        frame = pd.DataFrame(metrics.get("per_sample") or [])
        if frame.empty:
            continue
        frame.insert(0, "Mode", mode)
        per_sample_frames.append(frame)

    if per_sample_frames:
        combined = pd.concat(per_sample_frames, ignore_index=True)
        destination = Path("data/evaluation/per_sample.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination, index=False)
        logger.info(
            "Per-sample scores for %d mode(s) saved to %s",
            len(per_sample_frames),
            destination,
        )

        # Surface the worst questions per metric, so regressions are visible
        # without opening the CSV.
        # RAGAS names the question column `user_input`, not `question`.
        question_column = next(
            (c for c in ("user_input", "question") if c in combined.columns), None
        )
        metric_columns = [
            c for c in combined.columns
            if c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        ]
        for column in metric_columns:
            columns = [c for c in ("Mode", question_column, column) if c]
            scores = combined[columns].dropna(subset=[column]).sort_values(column).head(3)
            if scores.empty:
                continue
            print(f"\nLowest {column}:")
            for _, row in scores.iterrows():
                question = row[question_column] if question_column else ""
                print(f"  {row[column]:.3f}  [{row['Mode']}] {str(question)[:70]}")
    else:
        logger.warning("No per-sample scores were produced.")


if __name__ == "__main__":
    main()