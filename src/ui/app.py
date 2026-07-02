# app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
os.environ["OLLAMA_USE_CUDA"] = "0"  # Force CPU

import streamlit as st
import logging
from src.config.config import CONFIG
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import Generator
import psycopg2
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_pipeline():
    """Load the entire RAG pipeline once and cache it."""
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

    # 3. Retrievers
    dense = DenseRetriever(embedder, indexer)
    sparse = SparseRetriever(db_conn=conn, config=CONFIG)
    hybrid = HybridRetriever(dense, sparse)

    # 4. Generator
    generator = Generator(hybrid)

    logger.info("RAG pipeline loaded successfully.")
    return generator


def main():
    st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")
    st.title("📚 RAG Knowledge Assistant")
    st.markdown("Ask a question about your knowledge base. The system retrieves relevant context and generates a grounded answer with citations.")

    # Load the pipeline (cached)
    generator = load_pipeline()

    # Query input
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the JWST launch date?")
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit", type="primary", use_container_width=True)

    if submit and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = generator.run(query, top_k=5)
                answer = result["answer"]
                sources = result["sources"]

                # Display answer
                st.subheader("Answer")
                st.markdown(f"**{answer}**")

                # Display sources (citations)
                st.subheader("Sources")
                if sources:
                    for doc_id in sources:
                        st.markdown(f"- 📄 `{doc_id}`")
                else:
                    st.info("No sources were used (the model may have answered without context).")

                # Optional: Show retrieved context in expander
                with st.expander("🔍 Show retrieved chunks (for debugging)"):
                    # To show context, we'd need to store it in the result. 
                    # For now, we can just show a message.
                    st.info("To display full context, modify `Generator.run()` to return the retrieved docs.")
                    st.write("You can extend this in Week 7.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("UI generation failed")
    elif submit and not query.strip():
        st.warning("Please enter a question.")

    # Footer
    st.divider()
    st.caption("Powered by RAG | Hybrid Retrieval | Ollama")


if __name__ == "__main__":
    main()