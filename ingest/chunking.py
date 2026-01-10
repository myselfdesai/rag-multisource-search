"""Chunking strategies for different document types."""
from typing import List, Dict, Any
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode


def chunk_pdf(text: str, chunks_data: List[Dict[str, Any]], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Chunk PDF text with page-aware splitting.
    
    Args:
        text: Full document text
        chunks_data: List of page-level chunks with metadata
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap size in tokens
    
    Returns:
        List of chunked documents with metadata
    """
    llama_docs = []
    for chunk_data in chunks_data:
        cleaned_text = chunk_data['text']
        page = chunk_data['page']
        
        doc = LlamaDocument(
            text=cleaned_text,
            metadata={
                'page': page,
                'source_type': 'pdf',
                'content_kind': 'paragraph'
            }
        )
        llama_docs.append(doc)
    
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    nodes = splitter.get_nodes_from_documents(llama_docs)
    
    result_chunks = []
    for idx, node in enumerate(nodes):
        page = node.metadata.get('page', 1)
        result_chunks.append({
            'text': node.text,
            'metadata': {
                **node.metadata,
                'chunk_index': idx,
                'chunking_strategy': f'pdf_{chunk_size}_{chunk_overlap}_recursive'
            },
            'page': page
        })
    
    return result_chunks


def chunk_docx(text: str, chunks_data: List[Dict[str, Any]], chunk_size: int = 750, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Chunk DOCX text with section-aware splitting.
    
    Args:
        text: Full document text
        chunks_data: List of paragraph-level chunks with metadata
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap size in tokens
    
    Returns:
        List of chunked documents with metadata
    """
    llama_docs = []
    for chunk_data in chunks_data:
        cleaned_text = chunk_data['text']
        section = chunk_data.get('section', 'Unknown')
        section_index = chunk_data.get('section_index', 0)
        
        doc = LlamaDocument(
            text=cleaned_text,
            metadata={
                'section': section,
                'section_index': section_index,
                'source_type': 'docx',
                'content_kind': 'paragraph'
            }
        )
        llama_docs.append(doc)
    
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    nodes = splitter.get_nodes_from_documents(llama_docs)
    
    result_chunks = []
    for idx, node in enumerate(nodes):
        section = node.metadata.get('section', 'Unknown')
        section_index = node.metadata.get('section_index', 0)
        result_chunks.append({
            'text': node.text,
            'metadata': {
                **node.metadata,
                'chunk_index': idx,
                'chunking_strategy': f'docx_{chunk_size}_{chunk_overlap}_recursive'
            },
            'section': section,
            'section_index': section_index
        })
    
    return result_chunks


def chunk_csv(chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk CSV data - one chunk per row, no overlap.
    
    Args:
        chunks_data: List of row-level chunks with metadata
    
    Returns:
        List of chunked documents with metadata
    """
    result_chunks = []
    for idx, chunk_data in enumerate(chunks_data):
        row_id = chunk_data['row_id']
        result_chunks.append({
            'text': chunk_data['text'],
            'metadata': {
                'row_id': row_id,
                'source_type': 'csv',
                'content_kind': 'row',
                'chunk_index': idx,
                'chunking_strategy': 'csv_per_row'
            },
            'row_id': row_id
        })
    
    return result_chunks


def chunk_document(source_type: str, text: str, chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk a document based on its source type.
    
    Args:
        source_type: 'pdf', 'docx', or 'csv'
        text: Full document text
        chunks_data: Raw chunks from loader
    
    Returns:
        List of chunked documents with metadata
    """
    if source_type == 'pdf':
        return chunk_pdf(text, chunks_data)
    elif source_type == 'docx':
        return chunk_docx(text, chunks_data)
    elif source_type == 'csv':
        return chunk_csv(chunks_data)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

