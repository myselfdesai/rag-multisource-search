"""Document loaders for PDF, DOCX, and CSV files."""
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
from docx import Document
import pandas as pd


def generate_doc_id(filepath: str, content_hash: str = None) -> str:
    """Generate deterministic document ID from filepath and optional content hash."""
    from slugify import slugify
    
    filename = Path(filepath).stem
    slug = slugify(filename, lowercase=True)
    
    if content_hash:
        short_hash = content_hash[:8]
        return f"{slug}_{short_hash}"
    return slug


def load_pdf(filepath: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Load PDF file and extract text with page metadata.
    
    Returns:
        Tuple of (full_text, chunks_with_metadata) where each chunk has:
        - text: extracted text
        - page: page number (1-indexed)
        - metadata: dict with page info
    """
    chunks = []
    full_text_parts = []
    
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                full_text_parts.append(text)
                chunks.append({
                    'text': text,
                    'page': page_num,
                    'metadata': {
                        'page': page_num,
                        'source_type': 'pdf'
                    }
                })
    
    full_text = '\n\n'.join(full_text_parts)
    
    content_hash = hashlib.md5(full_text.encode()).hexdigest()
    doc_id = generate_doc_id(filepath, content_hash)
    
    return full_text, chunks, doc_id


def load_docx(filepath: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Load DOCX file and extract text with section metadata.
    
    Returns:
        Tuple of (full_text, chunks_with_metadata) where each chunk has:
        - text: extracted text
        - section: section/heading context
        - metadata: dict with section info
    """
    doc = Document(filepath)
    chunks = []
    full_text_parts = []
    current_section = "Introduction"
    section_index = 0
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        if para.style.name.startswith('Heading'):
            current_section = text
            section_index += 1
        
        full_text_parts.append(text)
        chunks.append({
            'text': text,
            'section': current_section,
            'section_index': section_index,
            'metadata': {
                'section': current_section,
                'section_index': section_index,
                'source_type': 'docx'
            }
        })
    
    full_text = '\n\n'.join(full_text_parts)
    
    content_hash = hashlib.md5(full_text.encode()).hexdigest()
    doc_id = generate_doc_id(filepath, content_hash)
    
    return full_text, chunks, doc_id


def load_csv(filepath: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Load CSV file and convert rows to text chunks.
    
    Returns:
        Tuple of (full_text, chunks_with_metadata) where each chunk has:
        - text: formatted row as "key: value | key: value | ..."
        - row_id: row index
        - metadata: dict with row info
    """
    df = pd.read_csv(filepath)
    chunks = []
    full_text_parts = []
    
    for idx, row in df.iterrows():
        row_parts = []
        for col, val in row.items():
            if pd.notna(val):
                row_parts.append(f"{col}: {val}")
        
        row_text = " | ".join(row_parts)
        full_text_parts.append(row_text)
        
        chunks.append({
            'text': row_text,
            'row_id': idx,
            'metadata': {
                'row_id': idx,
                'source_type': 'csv'
            }
        })
    
    full_text = '\n'.join(full_text_parts)
    
    content_hash = hashlib.md5(full_text.encode()).hexdigest()
    doc_id = generate_doc_id(filepath, content_hash)
    
    return full_text, chunks, doc_id


def detect_file_type(filepath: str) -> str:
    """Detect file type from extension."""
    ext = Path(filepath).suffix.lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    elif ext == '.csv':
        return 'csv'
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_document(filepath: str) -> Tuple[str, List[Dict[str, Any]], str, str, str]:
    """
    Load a document based on its file type.
    
    Returns:
        Tuple of (full_text, chunks_with_metadata, doc_id, source_type)
    """
    file_type = detect_file_type(filepath)
    source_name = Path(filepath).name
    
    if file_type == 'pdf':
        full_text, chunks, doc_id = load_pdf(filepath)
    elif file_type == 'docx':
        full_text, chunks, doc_id = load_docx(filepath)
    elif file_type == 'csv':
        full_text, chunks, doc_id = load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return full_text, chunks, doc_id, source_name, file_type

