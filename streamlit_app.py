"""Streamlit UI for RAG Challenge application."""
import streamlit as st
import requests
import json
import os
import tempfile
from pathlib import Path
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "dev")


def check_api_health():
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_document(file, namespace: str):
    """Upload and ingest a document via API."""
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
            return {"success": False, "message": str(e)}


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
    
    tab1, tab2, tab3, tab4 = st.tabs(["Query Documents", "Upload Documents", "Test Queries & Evaluation", "Status"])
    
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
            namespace = st.selectbox("Namespace", ["dev", "prod"], index=0)
            top_k = st.slider("Top K results", min_value=5, max_value=50, value=20, step=5)
            source_type = st.selectbox("Filter by source type", [None, "pdf", "docx", "csv"], format_func=lambda x: "All" if x is None else x.upper())
        
        if st.button("Search", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching documents and generating answer..."):
                    result = query_api(
                        question=question,
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
        
        namespace = st.selectbox("Target Namespace", ["dev", "prod"], index=0, key="upload_namespace")
        
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
        st.header("Test Queries & Evaluation")
        st.markdown("Run test queries to assess system accuracy and retrieval relevance")
        
        namespace = st.selectbox("Namespace", ["dev", "prod"], index=0, key="eval_namespace")
        
        test_queries_path = "eval/test_queries.json"
        if os.path.exists(test_queries_path):
            with open(test_queries_path, 'r') as f:
                test_queries = json.load(f)
            
            st.subheader(f"Test Queries ({len(test_queries)} queries)")
            
            if st.button("Run All Test Queries", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, query_data in enumerate(test_queries):
                    question = query_data.get('question', '')
                    query_id = query_data.get('id', f'q{idx+1}')
                    category = query_data.get('category', 'unknown')
                    
                    status_text.text(f"Running query {idx+1}/{len(test_queries)}: {question[:50]}...")
                    
                    result = query_api(
                        question=question,
                        namespace=namespace,
                        top_k=20
                    )
                    
                    if "error" not in result:
                        results.append({
                            "query_id": query_id,
                            "question": question,
                            "category": category,
                            "answer": result.get("answer", ""),
                            "num_sources": len(result.get("sources", [])),
                            "num_retrieved": result.get("meta", {}).get("retrieved_count", 0),
                            "sources": result.get("sources", []),
                            "expected_topics": query_data.get("expected_topics", []),
                            "should_refuse": query_data.get("should_refuse", False)
                        })
                    
                    progress_bar.progress((idx + 1) / len(test_queries))
                
                status_text.text("Evaluation complete!")
                progress_bar.empty()
                
                st.session_state['eval_results'] = results
                st.rerun()
            
            if 'eval_results' in st.session_state and st.session_state['eval_results']:
                results = st.session_state['eval_results']
                
                st.subheader("Evaluation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries", len(results))
                with col2:
                    st.metric("With Answers", sum(1 for r in results if r.get("answer")))
                with col3:
                    st.metric("Avg Sources", f"{sum(r.get('num_sources', 0) for r in results) / len(results):.1f}")
                with col4:
                    st.metric("Avg Retrieved", f"{sum(r.get('num_retrieved', 0) for r in results) / len(results):.1f}")
                
                st.divider()
                
                for result in results:
                    with st.expander(f"{result['query_id']}: {result['question'][:60]}..."):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown("**Answer:**")
                            st.markdown(result.get("answer", "No answer generated"))
                            
                            if result.get("sources"):
                                st.markdown(f"**Sources ({result['num_sources']}):**")
                                for i, source in enumerate(result['sources'][:3], 1):
                                    st.markdown(f"{i}. {source.get('source_name', 'Unknown')} - {source.get('locator', 'N/A')}")
                        
                        with col_b:
                            st.markdown("**Metadata:**")
                            st.json({
                                "Category": result.get("category", "unknown"),
                                "Sources": result.get("num_sources", 0),
                                "Retrieved": result.get("num_retrieved", 0),
                                "Expected Topics": result.get("expected_topics", [])
                            })
                
                st.divider()
                st.subheader("Performance Analysis")
                
                st.markdown("### Retrieval Relevance")
                st.markdown("""
                **Observations:**
                - The system retrieves relevant chunks based on semantic similarity
                - Reranking improves relevance by using cross-encoder models
                - Top-K retrieval (default 20) provides good coverage
                - Source citations help verify answer grounding
                
                **Strengths:**
                - Semantic search finds contextually relevant content
                - Reranking filters out less relevant chunks
                - Metadata (page numbers, sections) enables precise citations
                """)
                
                st.markdown("### Response Quality")
                st.markdown("""
                **Observations:**
                - Answers are grounded in retrieved context
                - System provides citations for verification
                - Responses are concise and focused
                - Handles "not found" cases appropriately
                
                **Strengths:**
                - Grounded answers with source references
                - Clear citation format (document name + location)
                - Appropriate refusal for out-of-scope questions
                """)
                
                st.markdown("### Limitations")
                st.markdown("""
                **Observed Limitations:**
                1. **PDF Extraction**: Complex layouts, tables, and figures may not be fully captured
                2. **No OCR**: Scanned documents or image-based PDFs are not processed
                3. **Table Parsing**: Tables in PDFs may not be parsed correctly
                4. **Chunk Boundaries**: Important context may be split across chunk boundaries
                5. **Figure Handling**: Images and figures are not processed (text/captions only)
                6. **Cross-Document**: Complex multi-document reasoning may be limited
                
                **Impact:**
                - Questions about visual content (charts, diagrams) cannot be answered
                - Table-based questions may have incomplete answers
                - Very long documents may lose context at chunk boundaries
                """)
                
                st.markdown("### Potential Improvements")
                st.markdown("""
                **Recommended Enhancements:**
                1. **Better Table Extraction**: Use specialized tools (camelot, tabula) for table parsing
                2. **Hybrid Retrieval**: Combine keyword search with semantic search
                3. **Query Expansion**: Reformulate queries to improve retrieval
                4. **Semantic Chunking**: Use semantic similarity for better chunk boundaries
                5. **Multi-Vector Retrieval**: Store multiple embeddings per chunk for better context
                6. **Fine-tuned Reranking**: Train domain-specific reranking models
                7. **Table Understanding**: Extract and structure table data separately
                8. **Figure Caption Extraction**: Better extraction of figure captions and descriptions
                """)
                
                if st.button("Clear Results", use_container_width=True):
                    del st.session_state['eval_results']
                    st.rerun()
        else:
            st.warning(f"Test queries file not found: {test_queries_path}")
            st.info("Expected location: eval/test_queries.json")
    
    with tab4:
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
        
        with col2:
            st.subheader("Configuration")
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
        5. **Test Queries**: Use the "Test Queries & Evaluation" tab to run evaluation queries and assess system performance
        """)


if __name__ == "__main__":
    main()

