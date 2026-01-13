"""Streamlit UI for RAG Challenge application."""
import streamlit as st
import requests
import json
import os
import tempfile
from pathlib import Path
import time
from datetime import datetime
from io import BytesIO

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "prod")


def check_api_health():
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def validate_file_upload(file) -> tuple[bool, str]:
    """
    Validate uploaded file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.csv'}
    
    # Check file size
    file_size = file.size
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed size (50 MB)"
    
    # Check file extension
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '{file_ext}' not supported. Allowed types: PDF, DOCX, CSV"
    
    # Basic validation passed
    return True, ""


def upload_document(file, namespace: str):
    """Upload and ingest a document via API."""
    # Validate file first
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        return {"success": False, "message": error_msg}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            from ingest.ingest import ingest_document
            from storage.pinecone_store import PineconeStore
            
            store = PineconeStore(namespace=namespace)
            num_chunks = ingest_document(file_path, namespace, store, reingest=False)
            return {"success": True, "chunks": num_chunks, "message": f"Successfully ingested {num_chunks} chunks"}
        except Exception as e:
            return {"success": False, "message": f"Ingestion error: {str(e)}"}


def query_api(question: str, namespace: str, top_k: int = 20, source_type: str = None):
    """Query the RAG API."""
    payload = {
        "question": question,
        "namespace": namespace,
        "top_k": top_k,
        "source_type": source_type if source_type else None,
        "retrieval_only": False
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}




def main():
    st.set_page_config(
        page_title="RAG Challenge - Document Q&A",
        page_icon=None,
        layout="wide"
    )
    
    st.title("RAG Challenge - Document Q&A System")
    st.markdown("Retrieval-Augmented Generation for document question answering")
    
    if not check_api_health():
        st.error(f"FastAPI server is not running at {API_URL}")
        st.info("Please start the server with: `pipenv run python -m app.main`")
        st.stop()
    else:
        st.success("API server is running")
    
    tab1, tab2, tab3 = st.tabs(["Query Documents", "Upload Documents", "Status"])
    
    with tab1:
        st.header("Query Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What is the main topic of this document?",
                key="question_input"
            )
        
        with col2:
            namespace = "prod"  # Fixed to prod namespace
            top_k = st.slider("Top K results", min_value=5, max_value=50, value=20, step=5)
            source_type = st.selectbox("Filter by source type", [None, "pdf", "docx", "csv"], format_func=lambda x: "All" if x is None else x.upper())
        
        if st.button("Search", type="primary", use_container_width=True):
            question_stripped = question.strip()
            
            # Client-side validation
            if not question_stripped:
                st.warning("Please enter a question")
            elif len(question_stripped) < 3:
                st.warning("Question must be at least 3 characters long")
            elif len(question_stripped) > 2000:
                st.warning("Question is too long (maximum 2000 characters)")
            else:
                with st.spinner("Searching documents and generating answer..."):
                    result = query_api(
                        question=question_stripped,
                        namespace=namespace,
                        top_k=top_k,
                        source_type=source_type
                    )
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("Query completed!")
                    
                    st.subheader("Answer")
                    st.markdown(result.get("answer", "No answer generated"))
                    
                    sources = result.get("sources", [])
                    if sources:
                        st.subheader(f"Sources ({len(sources)})")
                        
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('source_name', 'Unknown')} - {source.get('locator', 'N/A')}"):
                                st.markdown(f"**Document ID:** `{source.get('doc_id', 'N/A')}`")
                                st.markdown(f"**Location:** {source.get('locator', 'N/A')}")
                                if source.get('similarity_score'):
                                    st.markdown(f"**Similarity Score:** {source.get('similarity_score', 0):.4f}")
                                st.markdown("**Snippet:**")
                                st.code(source.get('snippet', ''), language=None)
                    
                    meta = result.get("meta", {})
                    with st.expander("Query Metadata"):
                        st.json(meta)
    
    with tab2:
        st.header("Upload Documents")
        st.markdown("Upload PDF, DOCX, or CSV files to ingest into the vector store")
        
        namespace = "prod"  # Fixed to prod namespace
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "csv"],
            help="Supported formats: PDF, DOCX, CSV"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            if st.button("Upload & Ingest", type="primary", use_container_width=True):
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    result = upload_document(uploaded_file, namespace)
                
                if result.get("success"):
                    st.success(f"{result.get('message', 'Document ingested successfully')}")
                    if "chunks" in result:
                        st.info(f"Created {result['chunks']} chunks from the document")
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    with tab3:
        st.header("System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Health")
            if check_api_health():
                st.success("API server is healthy")
                try:
                    response = requests.get(f"{API_URL}/health", timeout=2)
                    health_data = response.json()
                    st.json(health_data)
                except:
                    pass
            else:
                st.error("API server is not responding")
                st.info("Start the server with: `pipenv run python -m app.main`")
        
        with col2:
            st.subheader("Configuration")
            try:
                from app.config import get_settings
                settings = get_settings()
                config_info = {
                    "API URL": API_URL,
                    "Default Namespace": DEFAULT_NAMESPACE,
                    "Embedding Model": settings.embedding_model,
                    "LLM Model": settings.llm_model,
                    "Top K": settings.default_top_k,
                    "Supported Formats": ["PDF", "DOCX", "CSV"]
                }
            except:
                config_info = {
                    "API URL": API_URL,
                    "Default Namespace": DEFAULT_NAMESPACE,
                    "Supported Formats": ["PDF", "DOCX", "CSV"]
                }
            st.json(config_info)
        
        st.subheader("Usage Instructions")
        st.markdown("""
        1. **Upload Documents**: Go to the "Upload Documents" tab and upload your PDF, DOCX, or CSV files
        2. **Query Documents**: Go to the "Query Documents" tab and ask questions about your uploaded documents
        3. **Filter Results**: Use the source type filter to search specific document types (PDF, DOCX, CSV)
        4. **View Sources**: Expand source citations to see where the answer came from
        5. **Evaluation Results**: View the latest evaluation report in the "Evaluation Results" tab
        """)


if __name__ == "__main__":
    main()

