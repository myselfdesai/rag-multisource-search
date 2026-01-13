import logging
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, get_response_synthesizer, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import get_settings
from rag.prompts import get_qa_prompt
from storage.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """RAG query engine with retrieval, reranking, and generation."""
    
    def __init__(self, namespace: str = "prod"):
        self.settings = get_settings()
        self.namespace = namespace
        self.store = PineconeStore(namespace=namespace)
        self.index = self.store.get_index()
        
        self.llm = OpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            temperature=0.0
        )
        
        self._query_engine = None
    
    def _build_retriever(
        self,
        top_k: int = 20,
        source_type: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> VectorIndexRetriever:
        """Build retriever with optional filters."""
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        if source_type or doc_id:
            class MetadataFilter(BaseNodePostprocessor):
                def __init__(self, source_type, doc_id):
                    super().__init__()
                    self.source_type = source_type
                    self.doc_id = doc_id
                
                def _postprocess_nodes(self, nodes, query_bundle):
                    filtered = []
                    for node in nodes:
                        metadata = node.metadata
                        if self.source_type and metadata.get('source_type') != self.source_type:
                            continue
                        if self.doc_id and metadata.get('doc_id') != self.doc_id:
                            continue
                        filtered.append(node)
                    
                    if len(filtered) < 3:
                        logger.warning(f"Filtered retrieval returned only {len(filtered)} results, using all results")
                        return nodes
                    
                    return filtered
            
            filter_postprocessor = MetadataFilter(source_type, doc_id)
            retriever.node_postprocessors = [filter_postprocessor]
        
        return retriever
    
    def _build_reranker(self, top_n: int = 4):
        """Build reranker using cross-encoder."""
        try:
            from sentence_transformers import CrossEncoder
            from pydantic import Field, PrivateAttr
            
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            class SimpleReranker(BaseNodePostprocessor):
                top_n: int = Field(default=4)
                _model: Any = PrivateAttr(default=None)
                
                def model_post_init(self, __context: Any) -> None:
                    """Initialize private attribute after Pydantic model initialization."""
                    self._model = model
                
                def _postprocess_nodes(
                    self,
                    nodes: List[NodeWithScore],
                    query_bundle: Optional[QueryBundle] = None,
                ) -> List[NodeWithScore]:
                    """Rerank nodes using cross-encoder model."""
                    if not nodes or query_bundle is None or len(nodes) <= self.top_n:
                        return nodes
                    
                    query_text = query_bundle.query_str
                    pairs = [[query_text, node.node.get_content()] for node in nodes]
                    scores = self._model.predict(pairs)
                    
                    reranked = [
                        NodeWithScore(node=node.node, score=float(score))
                        for node, score in zip(nodes, scores)
                    ]
                    reranked.sort(key=lambda x: x.score or 0.0, reverse=True)
                    return reranked[:self.top_n]
            
            reranker = SimpleReranker(top_n=top_n)
            logger.info("Using SentenceTransformer reranker")
            return reranker
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}. Proceeding without reranking.")
            return None
    
    def query(
        self,
        question: str,
        top_k: int = 20,
        source_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        retrieval_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute query with retrieval and optional generation.
        
        Returns:
            Dict with answer, sources, retrieved chunks, and metadata
        """
        retriever = self._build_retriever(top_k=top_k, source_type=source_type, doc_id=doc_id)
        
        retrieved_nodes = retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
        
        reranker = self._build_reranker(top_n=self.settings.rerank_top_n)
        if reranker:
            query_bundle = QueryBundle(question)
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            logger.info(f"After reranking: {len(retrieved_nodes)} nodes")
        
        retrieved_nodes = retrieved_nodes[:self.settings.rerank_top_n]
        
        retrieved_chunks = []
        for node in retrieved_nodes:
            metadata = node.metadata
            retrieved_chunks.append({
                'id': node.node_id,
                'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'metadata': metadata,
                'score': getattr(node, 'score', None)
            })
        
        if retrieval_only:
            return {
                'answer': '',
                'sources': [],
                'retrieved': retrieved_chunks,
                'meta': {
                    'top_k': top_k,
                    'rerank_top_n': self.settings.rerank_top_n,
                    'namespace': self.namespace,
                    'retrieved_count': len(retrieved_nodes)
                }
            }
        
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
            text_qa_template=get_qa_prompt()
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[reranker] if reranker else []
        )
        
        response = query_engine.query(question)
        
        sources = []
        for node in retrieved_nodes[:self.settings.rerank_top_n]:
            metadata = node.metadata
            locator = self._get_locator(metadata)
            
            sources.append({
                'doc_id': metadata.get('doc_id', 'unknown'),
                'source_name': metadata.get('source_name', 'unknown'),
                'locator': locator,
                'snippet': node.text[:150] + "..." if len(node.text) > 150 else node.text,
                'chunk_index': metadata.get('chunk_index', 0),
                'similarity_score': getattr(node, 'score', None)
            })
        
        return {
            'answer': str(response),
            'sources': sources,
            'retrieved': retrieved_chunks,
            'meta': {
                'top_k': top_k,
                'rerank_top_n': self.settings.rerank_top_n,
                'namespace': self.namespace,
                'retrieved_count': len(retrieved_nodes)
            }
        }
    
    def _get_locator(self, metadata: Dict[str, Any]) -> str:
        """Extract locator string from metadata."""
        source_type = metadata.get('source_type', '')
        if source_type == 'pdf':
            return f"page {metadata.get('page', '?')}"
        elif source_type == 'docx':
            section = metadata.get('section', 'Unknown')
            return f"section: {section}"
        elif source_type == 'csv':
            return f"row {metadata.get('row_id', '?')}"
        return "unknown"

