"""Chunking strategies for different document types."""
from typing import List, Dict, Any


def chunk_pdf(text: str, chunks_data: List[Dict[str, Any]], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Semantic chunking for PDFs - splits by paragraphs and groups them.
    
    This approach:
    - Splits text by paragraph boundaries (double newlines)
    - Groups paragraphs together until reaching token limit
    - Preserves semantic context
    
    Args:
        text: Full document text
        chunks_data: List of page-level chunks with metadata
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap size in tokens
    
    Returns:
        List of chunked documents with metadata
    """
    result_chunks = []
    chunk_idx = 0
    
    for chunk_data in chunks_data:
        page_text = chunk_data['text']
        page = chunk_data['page']
        
        # Split by paragraph boundaries (double newlines indicate semantic breaks)
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            continue
        
        # Group paragraphs into semantic chunks
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            # Rough token estimate (1 token ≈ 4 characters)
            para_tokens = len(para) // 4
            
            # If adding this paragraph would exceed limit, save current chunk
            if current_length + para_tokens > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                result_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'page': page,
                        'source_type': 'pdf',
                        'content_kind': 'paragraph',
                        'chunk_index': chunk_idx,
                        'chunking_strategy': f'pdf_semantic_{chunk_size}'
                    },
                    'page': page
                })
                chunk_idx += 1
                
                # Start new chunk with overlap (keep last paragraph if small enough)
                if current_chunk and len(current_chunk[-1]) // 4 < chunk_overlap:
                    current_chunk = [current_chunk[-1], para]
                    current_length = len(current_chunk[-1]) // 4 + para_tokens
                else:
                    current_chunk = [para]
                    current_length = para_tokens
            else:
                current_chunk.append(para)
                current_length += para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            result_chunks.append({
                'text': chunk_text,
                'metadata': {
                    'page': page,
                    'source_type': 'pdf',
                    'content_kind': 'paragraph',
                    'chunk_index': chunk_idx,
                    'chunking_strategy': f'pdf_semantic_{chunk_size}'
                },
                'page': page
            })
            chunk_idx += 1
    
    return result_chunks


def chunk_docx(text: str, chunks_data: List[Dict[str, Any]], chunk_size: int = 750, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Semantic chunking for DOCX - groups paragraphs by sections/headings.
    
    This approach:
    - Groups paragraphs under the same heading together
    - Creates chunks that respect section boundaries
    - Preserves document structure and hierarchy
    
    Args:
        text: Full document text
        chunks_data: List of paragraph-level chunks with metadata
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap size in tokens
    
    Returns:
        List of chunked documents with metadata
    """
    result_chunks = []
    chunk_idx = 0
    
    # Group paragraphs by section
    current_section = None
    current_section_index = 0
    section_paragraphs = []
    
    for chunk_data in chunks_data:
        para_text = chunk_data['text']
        section = chunk_data.get('section', 'Unknown')
        section_index = chunk_data.get('section_index', 0)
        
        # If we've moved to a new section, process the previous section
        if section != current_section and section_paragraphs:
            # Chunk the accumulated paragraphs for this section
            section_chunks = _chunk_section_paragraphs(
                section_paragraphs,
                current_section,
                current_section_index,
                chunk_size,
                chunk_overlap,
                chunk_idx
            )
            result_chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
            section_paragraphs = []
        
        current_section = section
        current_section_index = section_index
        section_paragraphs.append(para_text)
    
    # Don't forget the last section
    if section_paragraphs:
        section_chunks = _chunk_section_paragraphs(
            section_paragraphs,
            current_section,
            current_section_index,
            chunk_size,
            chunk_overlap,
            chunk_idx
        )
        result_chunks.extend(section_chunks)
    
    return result_chunks


def _chunk_section_paragraphs(
    paragraphs: List[str],
    section: str,
    section_index: int,
    chunk_size: int,
    chunk_overlap: int,
    start_idx: int
) -> List[Dict[str, Any]]:
    """Helper to chunk paragraphs within a section."""
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = start_idx
    
    for para in paragraphs:
        para_tokens = len(para) // 4  # Rough token estimate
        
        # If adding this paragraph would exceed limit, save current chunk
        if current_length + para_tokens > chunk_size and current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'section': section,
                    'section_index': section_index,
                    'source_type': 'docx',
                    'content_kind': 'paragraph',
                    'chunk_index': chunk_idx,
                    'chunking_strategy': f'docx_semantic_{chunk_size}'
                },
                'section': section,
                'section_index': section_index
            })
            chunk_idx += 1
            
            # Start new chunk with overlap
            if current_chunk and len(current_chunk[-1]) // 4 < chunk_overlap:
                current_chunk = [current_chunk[-1], para]
                current_length = len(current_chunk[-1]) // 4 + para_tokens
            else:
                current_chunk = [para]
                current_length = para_tokens
        else:
            current_chunk.append(para)
            current_length += para_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'section': section,
                'section_index': section_index,
                'source_type': 'docx',
                'content_kind': 'paragraph',
                'chunk_index': chunk_idx,
                'chunking_strategy': f'docx_semantic_{chunk_size}'
            },
            'section': section,
            'section_index': section_index
        })
    
    return chunks


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

