# Retrieval-Augmented Generation System

End-to-end RAG prototype built with LlamaIndex, Pinecone, and FastAPI for document-based question answering.

## Architecture

The system consists of three main components:

1. **Ingestion Pipeline**: Loads PDF, DOCX, and CSV documents, cleans and chunks them, generates embeddings, and stores them in Pinecone
2. **Retrieval Component**: Uses embeddings-based similarity search with reranking to fetch relevant document chunks
3. **Generation Component**: Synthesizes answers using an LLM (OpenAI) grounded in retrieved content

### Key Features

- Support for multiple document types (PDF, DOCX, CSV)
- Deterministic document and chunk IDs
- Rich metadata tracking (page numbers, sections, row IDs)
- Namespace-based separation (dev/prod)
- Reranking for improved retrieval quality
- Grounded answers with citations
- FastAPI web service
- Evaluation framework

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
   - **Answer Similarity**: Semantic similarity to ground truth
4. Save results to `eval/results/`:
   - `latest_results.json` - Detailed per-query results
   - `detailed_results_TIMESTAMP.json` - Timestamped backup
   - `summary_TIMESTAMP.csv` - Overall metrics summary
   - `per_query_TIMESTAMP.csv` - Per-query metrics table

View results in the Streamlit UI "Evaluation Results" tab.




## Deployment

### Local Development

The FastAPI app runs on `http://0.0.0.0:8000` by default. For production deployment:

1. Use a production ASGI server like Gunicorn with Uvicorn workers:
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. Set up environment variables in your deployment environment

3. Configure reverse proxy (nginx, etc.) if needed

4. Ensure Pinecone index exists and is accessible from your deployment environment

### Docker (Optional)

Example Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
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


