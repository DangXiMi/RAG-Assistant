import os
import time
import streamlit as st
import requests

# --- Configuration & Constants ---
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
QUERY_URL = f"{BASE_URL}/api/v1/query"
INGEST_URL = f"{BASE_URL}/api/v1/ingest"
STATUS_URL = f"{BASE_URL}/api/v1/job"

# Max limits for adaptive polling
POLLING_TIMEOUT_SECONDS = 120
POLLING_INTERVAL_SECONDS = 2

# Page setup
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- Sidebar Layout ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Manage your knowledge base and settings here.")
    st.markdown("---")
    
    st.subheader("📤 Upload Knowledge Base")
    
    # Form submission acts as the boundary
    with st.form("upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "html", "txt"],
            accept_multiple_files=False,
            key=f"uploader_{st.session_state.uploader_key}"
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
st.title("🤖 My RAG Assistant")
st.markdown("Ask questions based on your uploaded documents. Powered by Hybrid Search.")
st.markdown("---")

# Display previous chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

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
                "mode": "Hybrid"
            }
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    message_placeholder.markdown(answer)
                    if sources:
                        with sources_placeholder.expander("📚 View Sources"):
                            for source in sources:
                                st.markdown(f"- {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    message_placeholder.error(f"Backend Error: {response.text}")
            except requests.exceptions.ConnectionError:
                message_placeholder.error("Could not connect to the backend server. Is FastAPI running?")