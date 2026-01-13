# RAG System - Document Q&A

Production-ready RAG system built with LlamaIndex, Pinecone, FastAPI, and Streamlit.

## Features

- Multi-format support: PDF, DOCX, CSV
- Semantic search: Vector search + reranking for high precision
- Grounded answers: LLM responses with source citations
- Quality metrics: RAGAS evaluation framework
- Docker ready: One-command deployment
- Production: Health checks, monitoring, nginx reverse proxy

## Architecture

**4 Layers:**
1. **Ingestion**: Load → Clean → Chunk → Embed → Store (Pinecone)
2. **Retrieval**: Vector search (top-20) → Rerank (top-4)
3. **Generation**: LLM synthesis with citations
4. **Deployment**: FastAPI + Streamlit + nginx

**Design:**
- Scalable: Pinecone serverless, multi-worker FastAPI
- Modular: Pluggable loaders, chunkers, models
- Maintainable: Metadata tracking, RAGAS metrics, Docker

## Quick Start (Docker)

```bash
# Clone and setup
git clone <repo-url> cedar-rag
cd cedar-rag
cp .env.example .env  # Add your API keys

# Run
docker-compose up -d

# Access
open http://localhost
```

Services:
- `http://localhost` - Streamlit UI
- `http://localhost/api` - FastAPI backend

## Environment Variables

Required in `.env`:
```bash
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
PINECONE_ENVIRONMENT=us-east-1
OPENAI_API_KEY=your_key
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## Local Development (No Docker)

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env

# Run API
python -m app.main

# Run UI (in another terminal)
streamlit run streamlit_app.py
```

Access:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## AWS EC2 Deployment

```bash
# Launch Ubuntu 22.04 instance (t3.medium+)
# Security Group: Allow ports 22, 80

# SSH into instance
ssh ubuntu@<EC2-IP>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker ubuntu
# Log out and back in

# Deploy
git clone <repo-url> cedar-rag && cd cedar-rag
# Add .env with your credentials
docker-compose up -d

# Access
http://<EC2-IP>
```

## Usage

### Upload Documents
Use Streamlit UI to upload PDF, DOCX, or CSV files

### Query Documents
- **UI**: Enter questions in Streamlit
- **API**:
  ```bash
  curl -X POST http://localhost/api/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Your question here?"}'
  ```

### Evaluate System
```bash
python -m eval.ragas_eval
```
View results in UI "Evaluation Results" tab

## System Details

### Chunking Strategies
- **PDF**: 900 tokens, 120 overlap, paragraph-based (semantic chunking)
- **DOCX**: 750 tokens, 100 overlap, section-aware (semantic chunking)
- **CSV**: One chunk per row ( **Basic implementation** - see Future Improvements)

### Metadata Tracked
Each chunk includes: doc_id, source_name, source_type, page/section/row, chunk_index, embedding_model, timestamps

### Vector IDs
Format: `{doc_id}:{locator}:{content_kind}:{chunk_index}`
- PDF: `p5` (page 5)
- DOCX: `s2` (section 2)  
- CSV: `r10` (row 10)

## Future Improvements

### CSV Handling Enhancements
Current CSV implementation is basic (one chunk per row). Planned improvements:

1. **Intelligent Chunking**
   - Group related rows together based on content similarity
   - Support for large datasets (batch chunking)
   - Configurable row grouping strategies

2. **Header & Column Analysis**
   - Automatic header detection and handling
   - Column importance/weighting based on data types
   - Primary key/ID column identification

3. **Semantic Grouping**
   - Detect categorical groupings (e.g., group rows by department, category)
   - Temporal grouping (e.g., group time-series data by date ranges)
   - Hierarchical relationships (parent-child rows)

4. **Metadata Extraction**
   - Extract column names, data types, and statistics
   - Add column descriptions/context to embeddings
   - Support for multi-value columns (lists, arrays)

5. **Query Optimization**
   - Column-specific search (e.g., "find rows where status=active")
   - Aggregate queries (e.g., "summarize sales by region")
   - Hybrid search combining structured filters + semantic search

> [!NOTE]
> These improvements will be implemented in future releases. Current CSV support is functional but optimized for simple, small datasets.

## Troubleshooting

**Missing env vars?** Check `.env` file  
**Empty answers?** Verify documents uploaded and namespace matches  
**Docker issues?** Run `docker-compose logs -f`  

## Monitoring (Docker)

```bash
# Logs
docker-compose logs -f

# Status
docker-compose ps

# Resources
docker stats

# Restart
docker-compose restart

# Update
git pull && docker-compose down && docker-compose build && docker-compose up -d
```

## Repository Structure

```
cedar-rag/
├── app/              # FastAPI backend
├── rag/              # Query engine
├── ingest/           # Document loaders & chunking
├── storage/          # Pinecone integration
├── eval/             # RAGAS evaluation
├── streamlit_app.py  # Frontend UI
├── docker-compose.yml
└── requirements.txt
```
