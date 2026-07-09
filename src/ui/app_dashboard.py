# app_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAG System Evaluation Dashboard")
st.markdown("Monitor retrieval and generation quality across different strategies.")

# Check if metrics CSV exists
metrics_path = Path("data/evaluation/metrics.csv")
if not metrics_path.exists():
    st.warning("No evaluation results found. Please run `python scripts/run_evaluation.py` first.")
    st.stop()

# Load data
df = pd.read_csv(metrics_path, index_col=0)
st.sidebar.header("📁 Data Source")
st.sidebar.info(f"Loaded {len(df)} retrieval modes.")

# Sidebar: Mode selection
st.sidebar.header("⚙️ Filter")
selected_mode = st.sidebar.selectbox("Select Retrieval Mode", df.index.tolist())

# --- Main Dashboard ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Aggregated Metrics")
    # Plot metrics for selected mode as a bar chart
    metrics_row = df.loc[selected_mode]
    metrics_to_plot = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    plot_df = pd.DataFrame({
        "Metric": metrics_to_plot,
        "Score": [metrics_row.get(m, 0.0) for m in metrics_to_plot]
    })
    fig = px.bar(
        plot_df,
        x="Metric",
        y="Score",
        title=f"{selected_mode} Performance",
        color="Metric",
        range_y=[0, 1],
        text_auto=".2f",
        template="simple_white"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Summary Table")
    # Display the full aggregated table
    st.dataframe(
        df.style.background_gradient(cmap="Blues", subset=["faithfulness", "answer_relevancy", "context_precision", "context_recall"]),
        use_container_width=True,
        height=400
    )

# --- Comparison Section ---
st.divider()
st.subheader("🔍 Mode Comparison")
col1, col2 = st.columns(2)

with col1:
    # Select two modes to compare
    modes = df.index.tolist()
    mode_a = st.selectbox("Mode A", modes, index=0)
    mode_b = st.selectbox("Mode B", modes, index=min(1, len(modes)-1))

with col2:
    # Side‑by‑side bar chart
    compare_df = df.loc[[mode_a, mode_b]].reset_index().melt(
        id_vars=["Mode"],
        value_vars=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        var_name="Metric",
        value_name="Score"
    )
    fig2 = px.bar(
        compare_df,
        x="Metric",
        y="Score",
        color="Mode",
        barmode="group",
        title=f"{mode_a} vs {mode_b}",
        range_y=[0, 1],
        text_auto=".2f",
        template="simple_white"
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

# --- Per‑Sample Details ---
st.divider()
st.subheader(f"📋 Per‑Sample Evaluation – {selected_mode}")
st.markdown("Expand to see detailed scores for each question.")

# Note: The evaluator currently returns per_sample data, but we're not saving it.
# For this version, we'll rely on the aggregated CSV. 
# To get per-sample, we'd need to modify the evaluator to save per_sample as well.
# For now, we'll show a placeholder.
st.info(
    "Per‑sample scores are not yet saved. To enable this, modify `RAGASEvaluator.evaluate()` to save the `per_sample` dict to a CSV file."
)
st.dataframe(df)

# Footer
st.divider()
st.caption("Built with Streamlit | RAGAS Metrics ")