"""Pinecone vector store integration."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import get_settings

logger = logging.getLogger(__name__)


class PineconeStore:
    """Wrapper for Pinecone vector store operations."""
    
    def __init__(self, namespace: str = "dev"):
        self.settings = get_settings()
        self.namespace = namespace
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.embedding_model = OpenAIEmbedding(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key
        )
        self._ensure_index()
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(self.settings.pinecone_index_name),
            text_key="text",
            namespace=namespace
        )
    
    def _ensure_index(self):
        """Ensure Pinecone index exists, create if not."""
        index_name = self.settings.pinecone_index_name
        
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            dimension = self._get_embedding_dimension()
            logger.info(f"Creating Pinecone index '{index_name}' with dimension {dimension}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment
                )
            )
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'")
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the configured model."""
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return model_dims.get(self.settings.embedding_model, 1536)
    
    def generate_vector_id(self, doc_id: str, locator: str, content_kind: str, chunk_index: int) -> str:
        """Generate deterministic vector ID."""
        return f"{doc_id}:{locator}:{content_kind}:{chunk_index}"
    
    def generate_locator(self, source_type: str, chunk_data: Dict[str, Any]) -> str:
        """Generate locator string based on source type."""
        if source_type == 'pdf':
            page = chunk_data.get('page', 1)
            return f"p{page}"
        elif source_type == 'docx':
            section_index = chunk_data.get('section_index', 0)
            return f"s{section_index}"
        elif source_type == 'csv':
            row_id = chunk_data.get('row_id', 0)
            return f"r{row_id}"
        else:
            return "u0"
    
    def upsert_chunks(
        self,
        doc_id: str,
        source_name: str,
        source_type: str,
        chunks: List[Dict[str, Any]],
        namespace: str,
        ingestion_version: Optional[str] = None
    ) -> int:
        """
        Upsert chunks to Pinecone.
        
        Returns:
            Number of chunks successfully upserted
        """
        if ingestion_version is None:
            ingestion_version = datetime.utcnow().isoformat()[:19]
        
        vectors_to_upsert = []
        
        for chunk in chunks:
            chunk_index = chunk['metadata']['chunk_index']
            locator = self.generate_locator(source_type, chunk)
            vector_id = self.generate_vector_id(doc_id, locator, chunk['metadata']['content_kind'], chunk_index)
            
            text = chunk['text']
            
            metadata = {
                'doc_id': doc_id,
                'source_name': source_name,
                'source_type': source_type,
                'content_kind': chunk['metadata']['content_kind'],
                'chunk_index': chunk_index,
                'text': text,
                'ingestion_version': ingestion_version,
                'chunking_strategy': chunk['metadata']['chunking_strategy'],
                'embedding_model': self.settings.embedding_model,
                'created_at': datetime.utcnow().isoformat()
            }
            
            if source_type == 'pdf':
                metadata['page'] = chunk.get('page', 1)
            elif source_type == 'docx':
                metadata['section'] = chunk.get('section', 'Unknown')
                metadata['section_index'] = chunk.get('section_index', 0)
            elif source_type == 'csv':
                metadata['row_id'] = chunk.get('row_id', 0)
            
            embedding = self.embedding_model.get_text_embedding(text)
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        index = self.pc.Index(self.settings.pinecone_index_name)
        
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            logger.info(f"Upserted batch of {len(batch)} vectors to namespace '{namespace}'")
        
        logger.info(f"Successfully upserted {total_upserted} chunks for document '{source_name}'")
        return total_upserted
    
    def get_index(self) -> VectorStoreIndex:
        """Get or create VectorStoreIndex for querying."""
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context,
            embed_model=self.embedding_model
        )

