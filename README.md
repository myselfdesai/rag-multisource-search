# Retrieval-Augmented Generation System

End-to-end RAG prototype built with LlamaIndex, Pinecone, and FastAPI for document-based question answering.

## System Architecture

This RAG system is designed with four key layers for scalability, modularity, and maintainability:

### 1. Ingestion Layer
The ingestion pipeline processes documents and stores them as vector embeddings in Pinecone:

- **Document Loading**: Type-specific loaders for PDF, DOCX, and CSV files extract text with location metadata (pages, sections, rows)
- **Text Processing**: Normalization and cleaning (Unicode handling, whitespace, newlines)
- **Semantic Chunking**: Document-type-optimized strategies maintain semantic coherence:
  - PDF: Paragraph-based chunking (~900 tokens) with recursive sentence splitting
  - DOCX: Section-aware chunking (~750 tokens) grouped by headings
  - CSV: Row-level chunking (one row per chunk)
- **Embedding Generation**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector Storage**: Deterministic vector IDs enable idempotent re-ingestion; rich metadata supports filtering

### 2. Retrieval Layer
The retrieval component finds relevant chunks using hybrid search:

- **Vector Search**: Pinecone similarity search with configurable `top_k` (default: 20)
- **Metadata Filtering**: Optional filters by namespace, source type, or document ID
- **Reranking**: Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) reranks top-20 to top-4 for precision

### 3. Generation Layer
The generation component synthesizes grounded answers:

- **LLM**: OpenAI `gpt-4o-mini` with custom QA prompt template
- **Context Assembly**: Top-ranked chunks provide grounding context
- **Response Synthesis**: Compact mode balances quality and token efficiency
- **Citation Tracking**: Source attribution with document names, locators, and snippets

### 4. Deployment Layer
Production-ready containerized deployment:

- **FastAPI Backend**: RESTful API for querying and health checks
- **Streamlit Frontend**: Interactive UI for document upload, querying, and evaluation results
- **Nginx Reverse Proxy**: Routes `/api/*` to FastAPI, `/*` to Streamlit with WebSocket support
- **Docker Compose**: Orchestrates all services with health checks and automatic restarts

### Design Considerations

**Scalability**
- Pinecone serverless backend scales automatically with query volume
- FastAPI supports multiple uvicorn workers for concurrent request handling
- Deterministic IDs prevent duplicate ingestion and enable incremental updates
- Namespace separation allows isolated dev/prod environments

**Modularity**
- Clear separation of concerns: ingestion, retrieval, generation, and serving
- Pluggable components: easy to swap embedding models, LLMs, or vector stores
- Type-specific loaders and chunking strategies extend easily to new document formats

**Maintainability**
- Comprehensive metadata tracking enables debugging and auditing
- RAGAS evaluation framework provides automated quality metrics
- Configuration via environment variables simplifies deployment
- Docker Compose simplifies local development and production deployments

### Key Features

- Support for multiple document types (PDF, DOCX, CSV)
- Deterministic document and chunk IDs
- Rich metadata tracking (page numbers, sections, row IDs)
- Namespace-based separation (prod)
- Reranking for improved retrieval quality
- Grounded answers with citations
- FastAPI web service
- Streamlit UI with evaluation dashboard

## Repository Structure

```
.
├── app/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic request/response models
│   └── config.py            # Configuration management
├── ingest/
│   ├── ingest.py            # CLI ingestion pipeline
│   ├── loaders.py           # PDF, DOCX, CSV loaders
│   ├── cleaning.py          # Text normalization
│   └── chunking.py          # Semantic chunking strategies
├── rag/
│   ├── query_engine.py      # Query engine with retrieval + generation
│   └── prompts.py           # Prompt templates
├── storage/
│   └── pinecone_store.py    # Pinecone integration
├── streamlit_app.py         # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index (will be created if it doesn't exist)
- `PINECONE_ENVIRONMENT`: Pinecone environment/region (e.g., `us-east-1`)
- `OPENAI_API_KEY`: Your OpenAI API key
- `EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `LLM_MODEL`: LLM model for generation (default: `gpt-4o-mini`)
- `DEFAULT_NAMESPACE`: Default namespace for documents (default: `prod`)


## Usage 

### Ingestion

Ingest documents into Pinecone:

```bash
python -m ingest.ingest --input_dir data/ --namespace prod --reingest false
```

Arguments:
- `--input_dir`: Directory containing documents to ingest
- `--namespace`: Pinecone namespace (default: `prod`)
- `--reingest`: Whether to re-ingest already processed documents (default: `false`)

The ingestion pipeline will:
1. Detect file types automatically
2. Extract text with location metadata (page numbers for PDFs, sections for DOCX, row IDs for CSV)
3. Clean and normalize text
4. Chunk documents with semantic chunking strategies:
   - PDF: Paragraph-based chunking (~900 token target) with semantic boundaries
   - DOCX: Section-aware chunking (~750 token target) grouping by headings
   - CSV: One chunk per row
5. Generate embeddings
6. Upsert to Pinecone with rich metadata

### Start FastAPI Server

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Start Streamlit UI (Optional)

For a web-based interface, you can use the Streamlit UI:

```bash
pipenv run streamlit run streamlit_app.py
```

The UI will be available at `http://localhost:8501` and provides:
- Document upload interface
- Query interface with results display
- Source citations and metadata
- System status

**Note**: The FastAPI server must be running for the Streamlit UI to work.

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the core idea behind transformers?",
    "namespace": "prod",
    "top_k": 20
  }'
```

Request fields:
- `question` (required): The question to answer
- `namespace` (optional): Pinecone namespace (default: `prod`)
- `top_k` (optional): Number of chunks to retrieve (default: 20)
- `source_type` (optional): Filter by source type (`pdf`, `docx`, `csv`)
- `doc_id` (optional): Filter by specific document ID
- `retrieval_only` (optional): If `true`, return only retrieval results without generation

Response includes:
- `answer`: Generated answer text
- `sources`: List of citations with document names, locators, and snippets
- `retrieved`: Full retrieval results for debugging
- `meta`: Metadata about the query execution

### Evaluation

Evaluate RAG system performance using RAGAS metrics:

```bash
python -m eval.ragas_eval
```

This will:
1. Load test queries from `eval/queries.json`
2. Query the RAG API for each test question
3. Compute RAGAS metrics:
   - **Faithfulness**: Answer grounded in retrieved context
   - **Answer Relevancy**: Answer addresses the question
   - **Context Precision**: Relevant chunks ranked higher
   - **Context Recall**: All relevant context retrieved
   - **Answer Correctness**: Similarity to ground truth
   
4. Save results to `eval/results/`:
   - `latest_results.json` - Detailed per-query results
   - `detailed_results_TIMESTAMP.json` - Timestamped backup
   - `summary_TIMESTAMP.csv` - Overall metrics summary
   - `per_query_TIMESTAMP.csv` - Per-query metrics table

View results in the Streamlit UI "Evaluation Results" tab.




## Deployment

### Local Docker Deployment

The easiest way to run the application is using Docker Compose:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Stop all services
docker-compose down
```

Services will be available at:
- **Streamlit UI**: `http://localhost` (nginx routes root to Streamlit)
- **FastAPI API**: `http://localhost/api` (nginx routes `/api/*` to FastAPI)

**Architecture:**
- `fastapi`: Backend API (port 8000)
- `streamlit`: Frontend UI (port 8501)
- `nginx`: Reverse proxy (port 80) - routes `/api/*` → FastAPI, `/*` → Streamlit

### AWS EC2 Deployment

For production deployment on AWS EC2:

#### 1. Launch EC2 Instance

- **Instance Type**: t3.medium or larger (2 vCPU, 4 GB RAM minimum)
- **OS**: Ubuntu 22.04 LTS
- **Security Group**: Allow inbound traffic on:
  - Port 22 (SSH)
  - Port 80 (HTTP)
  - Port 443 (HTTPS, optional for SSL)
- **Storage**: 20 GB EBS volume minimum
- **(Optional)** Attach Elastic IP for stable DNS

#### 2. Install Docker

SSH into your EC2 instance and install Docker:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Log out and log back in for group changes to take effect.

#### 3. Clone and Configure

```bash
git clone <your-repo-url> cedar-rag
cd cedar-rag

# Create and configure environment file
cp .env.example .env
nano .env  # Add your API keys and configuration
```

Required environment variables:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_ENVIRONMENT`
- `OPENAI_API_KEY`
- `EMBEDDING_MODEL=text-embedding-3-small`
- `LLM_MODEL=gpt-4o-mini`

#### 4. Deploy

```bash
docker-compose up -d
```

#### 5. Access Application

- Via EC2 Public IP: `http://<EC2-PUBLIC-IP>`

#### Monitoring

```bash
# View logs
docker-compose logs -f

# Check resource usage
docker stats

# Restart services
docker-compose restart

# Update and redeploy
git pull
docker-compose down
docker-compose build
docker-compose up -d
```

### Local Development (without Docker)

The FastAPI app runs on `http://0.0.0.0:8000` by default. 

```bash
python -m app.main
```



## Configuration Details

### Chunking Strategies

- **PDF**: 900 token chunks, 120 token overlap, recursive sentence splitting
- **DOCX**: 750 token chunks, 100 token overlap, section-aware splitting
- **CSV**: One chunk per row, no overlap

### Metadata Schema

Each chunk in Pinecone includes:
- `doc_id`: Deterministic document ID
- `source_name`: Original filename
- `source_type`: `pdf`, `docx`, or `csv`
- `content_kind`: `paragraph` or `row`
- `chunk_index`: Chunk sequence number
- `text`: Chunk text content
- `page` (PDF only): Page number
- `section` (DOCX only): Section/heading name
- `row_id` (CSV only): Row index
- `ingestion_version`: Timestamp of ingestion
- `chunking_strategy`: Strategy used for chunking
- `embedding_model`: Model used for embeddings
- `created_at`: Creation timestamp

### Vector ID Format

Deterministic vector IDs: `{doc_id}:{locator}:{content_kind}:{chunk_index}`

- PDF locator: `p{page}` (e.g., `p5`)
- DOCX locator: `s{section_index}` (e.g., `s2`)
- CSV locator: `r{row_id}` (e.g., `r10`)

## Troubleshooting

### Common Issues

1. **Missing environment variables**: Ensure all required variables are set in `.env`
2. **Pinecone index not found**: The index will be created automatically if it doesn't exist
3. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
4. **Low retrieval quality**: Try adjusting `top_k` or reranking parameters
5. **Empty answers**: Check if documents were ingested successfully and namespace matches

### Logging

The application uses Python's logging module. Set log level via environment variable:

```bash
export LOG_LEVEL=DEBUG
```


