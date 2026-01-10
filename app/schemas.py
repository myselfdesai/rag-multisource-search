"""Pydantic schemas for API request/response models."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="The question to answer")
    namespace: str = Field(default="dev", description="Pinecone namespace")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of chunks to retrieve")
    source_type: Optional[str] = Field(default=None, description="Filter by source type: pdf, docx, or csv")
    doc_id: Optional[str] = Field(default=None, description="Filter by specific document ID")
    retrieval_only: bool = Field(default=False, description="If true, return only retrieval results without generation")


class SourceCitation(BaseModel):
    """Citation information for a source chunk."""
    doc_id: str
    source_name: str
    locator: str = Field(..., description="Page number (pdf), section (docx), or row_id (csv)")
    snippet: str = Field(..., description="Short text snippet from the chunk")
    chunk_index: int
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer or empty if retrieval_only=true")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of source citations")
    retrieved: List[Dict[str, Any]] = Field(default_factory=list, description="Full retrieval results for debugging")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the query execution")

