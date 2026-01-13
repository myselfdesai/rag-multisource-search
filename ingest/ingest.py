"""CLI ingestion pipeline for loading, chunking, and indexing documents."""
import argparse
import logging
import os
from pathlib import Path
from typing import List

from ingest.loaders import load_document, detect_file_type
from ingest.cleaning import clean_text
from ingest.chunking import chunk_document
from storage.pinecone_store import PineconeStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_documents(input_dir: str) -> List[str]:
    """Find all supported documents in input directory."""
    supported_extensions = {'.pdf', '.docx', '.csv'}
    documents = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    for filepath in input_path.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
            documents.append(str(filepath))
    
    logger.info(f"Found {len(documents)} documents in {input_dir}")
    return documents


def ingest_document(
    filepath: str,
    namespace: str,
    store: PineconeStore,
    reingest: bool = False
) -> int:
    """
    Ingest a single document.
    
    Returns:
        Number of chunks ingested
    """
    logger.info(f"Processing: {filepath}")
    
    try:
        full_text, chunks_data, doc_id, source_name, source_type = load_document(filepath)
        logger.info(f"Loaded document: {source_name} (type: {source_type}, doc_id: {doc_id})")
        
        cleaned_text = clean_text(full_text)
        
        chunked_docs = chunk_document(source_type, cleaned_text, chunks_data)
        logger.info(f"Created {len(chunked_docs)} chunks")
        
        cleaned_chunks = []
        for chunk in chunked_docs:
            cleaned_chunk_text = clean_text(chunk['text'])
            if cleaned_chunk_text:
                chunk['text'] = cleaned_chunk_text
                cleaned_chunks.append(chunk)
        
        if not cleaned_chunks:
            logger.warning(f"No valid chunks after cleaning for {source_name}")
            return 0
        
        num_upserted = store.upsert_chunks(
            doc_id=doc_id,
            source_name=source_name,
            source_type=source_type,
            chunks=cleaned_chunks,
            namespace=namespace
        )
        
        logger.info(f"Successfully ingested {num_upserted} chunks for {source_name}")
        return num_upserted
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}", exc_info=True)
        return 0


def main():
    """CLI entry point for ingestion."""
    parser = argparse.ArgumentParser(description='Ingest documents into Pinecone vector store')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing documents to ingest')
    parser.add_argument('--namespace', type=str, default='prod', help='Pinecone namespace')
    parser.add_argument('--reingest', action='store_true', help='Re-ingest documents even if already indexed')
    
    args = parser.parse_args()
    
    logger.info(f"Starting ingestion: input_dir={args.input_dir}, namespace={args.namespace}")
    
    store = PineconeStore()
    
    documents = find_documents(args.input_dir)
    
    if not documents:
        logger.warning("No documents found to ingest")
        return
    
    total_chunks = 0
    for doc_path in documents:
        chunks = ingest_document(doc_path, args.namespace, store, args.reingest)
        total_chunks += chunks
    
    logger.info(f"Ingestion complete. Total chunks ingested: {total_chunks}")


if __name__ == '__main__':
    main()

