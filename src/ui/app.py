import os
import sys
import time
from pathlib import Path
import streamlit as st
import requests

# Allow the UI to run as `streamlit run src/ui/app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Single source of truth for accepted upload types, shared with the backend.
from src.ingestion.loaders import supported_extension_list  # noqa: E402

# --- Configuration & Constants ---
# BACKEND_URL wins if set; otherwise probe the ports this project actually uses.
# The old default (8000) silently pointed at nothing when the API was started
# on another port, and the UI just said "could not connect".
_DEFAULT_BACKENDS = [
    "http://localhost:8010",  # scripts start the API here
    "http://localhost:8000",  # docker compose / README default
]


def _resolve_backend() -> str:
    """Return a reachable backend URL, preferring $BACKEND_URL."""
    configured = os.getenv("BACKEND_URL")
    if configured:
        return configured.rstrip("/")

    for candidate in _DEFAULT_BACKENDS:
        try:
            response = requests.get(f"{candidate}/api/v1/health", timeout=2)
            if response.status_code == 200:
                return candidate
        except requests.exceptions.RequestException:
            continue

    return _DEFAULT_BACKENDS[0]


BASE_URL = _resolve_backend()
QUERY_URL = f"{BASE_URL}/api/v1/query"
INGEST_URL = f"{BASE_URL}/api/v1/ingest"
STATUS_URL = f"{BASE_URL}/api/v1/job"
HEALTH_URL = f"{BASE_URL}/api/v1/health"
FORMATS_URL = f"{BASE_URL}/api/v1/formats"

# Retrieval strategies exposed by the API, in the order they are offered.
RETRIEVAL_MODES = ["Hybrid", "HyDE", "Multi-Query", "Reranked"]
DEFAULT_MODE = "Hybrid"

# Extensions the loaders actually support (pdf, docx, xlsx, png, ...).
SUPPORTED_TYPES = supported_extension_list()

# Max limits for adaptive polling
POLLING_TIMEOUT_SECONDS = 300
POLLING_INTERVAL_SECONDS = 2

# Page setup
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


@st.cache_data(ttl=15, show_spinner=False)
def backend_health() -> dict:
    """Ask the API whether it is up, and whether OCR is usable.

    Cached briefly: Streamlit re-runs this whole script on every interaction,
    and probing the API each time would add latency to every keystroke.
    """
    try:
        response = requests.get(HEALTH_URL, timeout=4)
        if response.status_code != 200:
            return {"ok": False, "detail": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "detail": type(exc).__name__}

    info: dict = {"ok": True, "detail": ""}
    try:
        formats = requests.get(FORMATS_URL, timeout=4).json()
        ocr = formats.get("ocr", {})
        info["ocr"] = bool(ocr.get("available"))
        info["ocr_reason"] = ocr.get("reason")
        info["extensions"] = formats.get("extensions", [])
    except requests.exceptions.RequestException:
        pass
    return info

# --- Sidebar Layout ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Manage your knowledge base and settings here.")
    st.markdown("---")

    # Connection status. Without this, a misconfigured port looks like a crash.
    health = backend_health()
    st.subheader("🔌 Backend")
    if health["ok"]:
        st.success(f"Connected\n\n`{BASE_URL}`")
        if health.get("ocr") is False:
            st.warning(
                "OCR unavailable, so images and scanned PDFs cannot be read.\n\n"
                f"{health.get('ocr_reason') or ''}"
            )
        elif health.get("ocr"):
            st.caption("OCR available — images and scanned PDFs supported.")
    else:
        st.error(f"Not reachable: {BASE_URL}\n\n{health['detail']}")
        st.caption(
            "Start the API first:\n\n"
            "`uvicorn src.api.main:app --host 127.0.0.1 --port 8010`"
        )

    st.markdown("---")

    st.subheader("🔍 Retrieval Strategy")
    mode = st.selectbox(
        "How chunks are retrieved",
        RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_MODE),
        help=(
            "Hybrid fuses vector and keyword search (fastest, best default). "
            "HyDE and Multi-Query expand the question with the LLM first, so "
            "they are slower. Reranked applies a cross-encoder to reorder "
            "results for precision."
        ),
    )

    st.markdown("---")

    st.subheader("📤 Upload Knowledge Base")
    
    # Form submission acts as the boundary
    with st.form("upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=SUPPORTED_TYPES,
            accept_multiple_files=False,
            key=f"uploader_{st.session_state.uploader_key}"
        )
        st.caption(
            "Supported: PDF, Word, Excel, CSV, HTML, text and images "
            "(images and scanned PDFs are read with OCR)."
        )
        submit_upload = st.form_submit_button("Process Document")
        
    if submit_upload and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        status_macro = st.empty()
        status_macro.info(f"⏳ Initializing upload for {uploaded_file.name}...")
        
        try:
            response = requests.post(INGEST_URL, files=files)
            if response.status_code == 200:
                data = response.json()
                job_id = data["job_id"]
                
                # --- Adaptive Job Polling Loop ---
                start_time = time.time()
                status_url = f"{STATUS_URL}/{job_id}"
                is_complete = False
                
                while time.time() - start_time < POLLING_TIMEOUT_SECONDS:
                    status_resp = requests.get(status_url)
                    
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        current_status = status_data.get("status", "unknown")
                        
                        if current_status == "done":
                            chunks = status_data.get('result', {}).get('chunks', 0)
                            status_macro.success(f"✅ Indexed! {chunks} chunks created.")
                            is_complete = True
                            break
                        elif current_status == "failed":
                            error_msg = status_data.get("result", {}).get("error", "Unknown error")
                            status_macro.error(f"❌ Processing failed: {error_msg}")
                            is_complete = True
                            break
                        else:
                            # Show an updated time context so users know the app hasn't crashed
                            elapsed = int(time.time() - start_time)
                            status_macro.info(f"⏳ Processing... status: `{current_status}` ({elapsed}s elapsed)")
                    
                    elif status_resp.status_code == 404:
                        status_macro.warning("⏳ Job queued, waiting for worker pickup...")
                    else:
                        status_macro.error(f"⚠️ Unexpected status check response: {status_resp.status_code}")
                        
                    time.sleep(POLLING_INTERVAL_SECONDS)
                
                if not is_complete:
                    status_macro.error(f"❌ Ingestion timed out after {POLLING_TIMEOUT_SECONDS}s. Checking background task logs recommended.")
                
                # --- State Reset Trigger ---
                # Increment the key to completely strip the old file from the user's view
                st.session_state.uploader_key += 1
                time.sleep(1.5)  # Let the success message sit briefly before refreshing the layout
                st.rerun()
                
            else:
                status_macro.error(f"❌ Upload failed: {response.text}")
        except requests.exceptions.ConnectionError:
            status_macro.error("❌ Could not connect to the backend server. Is FastAPI running?")

# --- Main Chat UI Layout ---
st.title("🤖 RAG Knowledge Assistant")
st.markdown(
    "Ask questions about your uploaded documents. Answers are grounded in the "
    "retrieved passages and cite the file and page they came from."
)
st.caption(f"Strategy: **{mode}** · Backend: `{BASE_URL}`")
st.markdown("---")

def format_source(source) -> str:
    """Render one source reference for display.

    Handles the structured form (`{"source": file, "location": "p.3"}`) and the
    plain string form, so it works either way.
    """
    if isinstance(source, dict):
        name = source.get("source") or source.get("filename") or "unknown"
        location = source.get("location") or ""
        if not location:
            if source.get("page") is not None:
                location = f"p.{source['page']}"
            elif source.get("sheet"):
                location = f"sheet '{source['sheet']}'"
        reference = source.get("ref")
        prefix = f"**[{reference}]** " if reference else ""
        return f"{prefix}`{name}` — {location}" if location else f"{prefix}`{name}`"
    return str(source)


def render_sources(sources) -> None:
    """Show a de-duplicated source list, preserving order."""
    seen = []
    for source in sources:
        label = format_source(source)
        if label not in seen:
            seen.append(label)
    for label in seen:
        st.markdown(f"- {label}")


# Display previous chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                render_sources(msg["sources"])
        if msg.get("mode"):
            st.caption(f"strategy: {msg['mode']}")

# Chat Input field
if user_input := st.chat_input("Ask a question about your data..."):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            payload = {
                "question": user_input,
                "user_id": "user_123",
                "mode": mode
            }
            try:
                response = requests.post(QUERY_URL, json=payload, timeout=600)
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    # Prefer the structured references when present: they carry
                    # the filename and page separately, which reads better than
                    # a pre-joined "[1] file, p.3" string.
                    sources = data.get("source_refs") or data.get("sources", [])

                    message_placeholder.markdown(answer)

                    if not data.get("answered", True):
                        st.caption(
                            "No supporting passage was found in the indexed "
                            "documents, so the assistant declined to answer."
                        )

                    if sources:
                        with sources_placeholder.expander("📚 View Sources"):
                            render_sources(sources)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources,
                        "mode": mode,
                    })
                else:
                    message_placeholder.error(f"Backend Error: {response.text}")
            except requests.exceptions.ConnectionError:
                message_placeholder.error(
                    f"Could not connect to the backend at {BASE_URL}. "
                    "Is the FastAPI server running?"
                )
            except requests.exceptions.Timeout:
                message_placeholder.error(
                    "The backend took too long to answer. On CPU with a large "
                    "model this can happen; try the Hybrid strategy, which is "
                    "the fastest."
                )