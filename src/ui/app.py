import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/api/v1/query"

st.title("My RAG Assistant")

user_input = st.text_input("Ask a question about your data:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        # Format the payload to match FastAPI's Pydantic schema
        payload = {
            "question": user_input,
            "user_id": "user_123",
            "mode": "Hybrid"
        }
        
        try:
            # Send request to FastAPI backend
            response = requests.post(BACKEND_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Render the response
                st.write("### Answer:")
                st.write(data["answer"])
                
                st.write("### Sources:")
                for source in data["sources"]:
                    st.caption(f"- {source}")
            else:
                st.error(f"Backend Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Is FastAPI running?")