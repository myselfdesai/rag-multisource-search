import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import QueryRequest, QueryResponse, SourceCitation
from rag.query_engine import QueryEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Challenge API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
query_engines = {}


def get_query_engine(namespace: str) -> QueryEngine:
    """Get or create query engine for namespace."""
    if namespace not in query_engines:
        query_engines[namespace] = QueryEngine(namespace=namespace)
    return query_engines[namespace]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-challenge-api"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query endpoint for RAG-based question answering.
    """
    try:
        engine = get_query_engine(request.namespace)
        
        result = engine.query(
            question=request.question,
            top_k=request.top_k,
            source_type=request.source_type,
            doc_id=request.doc_id,
            retrieval_only=request.retrieval_only
        )
        
        sources = [
            SourceCitation(**src) for src in result['sources']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            retrieved=result['retrieved'],
            meta=result['meta']
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

