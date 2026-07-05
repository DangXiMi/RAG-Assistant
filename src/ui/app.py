# app.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import streamlit as st
import logging
import psycopg2
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
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
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


def main():
    st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")
    st.title("📚 RAG Knowledge Assistant")
    st.markdown("Ask a question about your knowledge base. The system retrieves relevant context and generates a grounded answer with citations.")

    # Load pipeline
    pipeline = load_pipeline()
    generator = pipeline["generator"]
    retrievers = pipeline["retrievers"]

    # Sidebar: Mode selection
    with st.sidebar:
        st.header("⚙️ Retrieval Mode")
        mode = st.selectbox(
            "Select retrieval strategy",
            list(retrievers.keys()),
            index=0,
            help="Choose the retrieval method to use for answering."
        )
        st.divider()
        st.caption("**Hybrid**: Dense + Sparse with RRF")
        st.caption("**HyDE**: Generate hypothetical document")
        st.caption("**Multi-Query**: Multiple query variants fused")
        st.caption("**Reranked**: Hybrid + Cross-encoder reranking")

    # Main query input
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the JWST launch date?")
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Submit", type="primary", use_container_width=True)

    if submit and query.strip():
        with st.spinner(f"Retrieving with {mode} and generating answer..."):
            try:
                # Get the selected retriever
                selected_retriever = retrievers[mode]

                # Run generation with the selected retriever
                result = generator.run(
                    query,
                    top_k=5,
                    retriever=selected_retriever
                )

                answer = result["answer"]
                sources = result["sources"]
                # The generator currently doesn't return the retrieved documents.
                # We'll extend it later; for now we can fetch them via retriever separately.
                # For debugging, we can optionally display them.

                # Display answer
                st.subheader("Answer")
                st.markdown(f"**{answer}**")

                # Display sources
                st.subheader("Sources")
                if sources:
                    for doc_id in sources:
                        st.markdown(f"- 📄 `{doc_id}`")
                else:
                    st.info("No sources were used.")

                # Debug expander: show retrieved chunks
                with st.expander("🔍 Show retrieved chunks (debug)"):
                    # To avoid extra retrieval calls, we could have generator return them.
                    # We'll quickly fetch them again using the selected retriever.
                    # For a more efficient approach, modify generator to return docs.
                    chunks = selected_retriever.search(query, top_k=5)
                    for i, doc in enumerate(chunks):
                        st.markdown(f"**Rank {i+1}** (score: `{doc['score']:.4f}`)")
                        st.markdown(f"ID: `{doc['id']}`")
                        st.markdown(f"Text: {doc['text'][:300]}...")
                        st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("UI generation failed")

    elif submit and not query.strip():
        st.warning("Please enter a question.")

    st.divider()
    st.caption("Powered by RAG | Hybrid Retrieval | Ollama")


if __name__ == "__main__":
    main()