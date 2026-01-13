from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(
        ..., 
        min_length=3,
        max_length=2000,
        description="The question to answer"
    )
    namespace: str = Field(default="prod", description="Pinecone namespace")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of chunks to retrieve")
    source_type: Optional[str] = Field(default=None, description="Filter by source type: pdf, docx, or csv")
    doc_id: Optional[str] = Field(default=None, description="Filter by specific document ID")
    retrieval_only: bool = Field(default=False, description="If true, return only retrieval results without generation")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate and sanitize question input."""
        # Strip and normalize whitespace
        v = v.strip()
        v = re.sub(r'\s+', ' ', v)
        
        # Check minimum length after normalization
        if len(v) < 3:
            raise ValueError("Question must be at least 3 characters long")
        
        # Check for prompt injection attempts
        suspicious_patterns = [
            r'ignore\s+(all\s+)?(previous\s+)?instructions?',
            r'ignore\s+above',
            r'disregard\s+(all\s+)?(previous\s+)?instructions?',
            r'system\s+prompt',
            r'you\s+are\s+now',
            r'jailbreak',
            r'roleplay\s+as',
            r'pretend\s+you\s+are',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Question contains suspicious content that violates input policy")
        
        return v
    
    @field_validator('namespace')
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace is set to prod."""
        if v != "prod":
            raise ValueError("Only 'prod' namespace is supported")
        return v
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate source type is one of the allowed values."""
        if v is not None and v not in ['pdf', 'docx', 'csv']:
            raise ValueError("Source type must be one of: pdf, docx, csv")
        return v
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate doc_id format."""
        if v is not None:
            # Sanitize doc_id to prevent injection
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Document ID contains invalid characters")
            if len(v) > 200:
                raise ValueError("Document ID is too long")
        return v


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


