"""Evaluation runner for test queries."""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvalRunner:
    """Runner for evaluating RAG system on test queries."""
    
    def __init__(self, api_url: str = "http://localhost:8000", namespace: str = "dev"):
        self.api_url = api_url
        self.namespace = namespace
        self.results = []
    
    def load_queries(self, queries_path: str) -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        with open(queries_path, 'r') as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} test queries")
        return queries
    
    def run_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query against the API."""
        question = query_data['question']
        query_id = query_data.get('id', 'unknown')
        
        logger.info(f"Running query {query_id}: {question[:50]}...")
        
        payload = {
            "question": question,
            "namespace": self.namespace,
            "top_k": 20,
            "retrieval_only": False
        }
        
        try:
            response = requests.post(f"{self.api_url}/query", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            return {
                "query_id": query_id,
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "retrieved": result.get("retrieved", []),
                "meta": result.get("meta", {}),
                "query_data": query_data
            }
        except Exception as e:
            logger.error(f"Error running query {query_id}: {e}")
            return {
                "query_id": query_id,
                "question": question,
                "answer": "",
                "sources": [],
                "retrieved": [],
                "meta": {},
                "error": str(e),
                "query_data": query_data
            }
    
    def format_result_for_scoring(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format result with fields for manual scoring."""
        retrieved_ids = [r.get('id', 'unknown') for r in result.get('retrieved', [])]
        retrieved_metadata = [
            {
                "id": r.get('id', 'unknown'),
                "doc_id": r.get('metadata', {}).get('doc_id', 'unknown'),
                "source_name": r.get('metadata', {}).get('source_name', 'unknown'),
                "locator": self._extract_locator(r.get('metadata', {})),
                "snippet": r.get('text', '')[:100]
            }
            for r in result.get('retrieved', [])
        ]
        
        return {
            "query_id": result.get("query_id"),
            "question": result.get("question"),
            "category": result.get("query_data", {}).get("category", "unknown"),
            "answer": result.get("answer", ""),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_metadata": retrieved_metadata,
            "num_retrieved": len(retrieved_ids),
            "num_sources": len(result.get("sources", [])),
            "retrieval_relevance": None,
            "groundedness": None,
            "completeness": None,
            "notes": "",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_locator(self, metadata: Dict[str, Any]) -> str:
        """Extract locator from metadata."""
        source_type = metadata.get('source_type', '')
        if source_type == 'pdf':
            return f"page {metadata.get('page', '?')}"
        elif source_type == 'docx':
            return f"section: {metadata.get('section', 'Unknown')}"
        elif source_type == 'csv':
            return f"row {metadata.get('row_id', '?')}"
        return "unknown"
    
    def run_evaluation(self, queries_path: str, output_path: str):
        """Run full evaluation and save results."""
        queries = self.load_queries(queries_path)
        
        logger.info(f"Running evaluation on {len(queries)} queries")
        
        for query_data in queries:
            result = self.run_query(query_data)
            formatted = self.format_result_for_scoring(result)
            self.results.append(formatted)
        
        os.makedirs(Path(output_path).parent, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(f"Total queries: {len(self.results)}")
        
        return self.results


def main():
    """CLI entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run evaluation on test queries')
    parser.add_argument('--queries', type=str, default='eval/test_queries.json', help='Path to test queries JSON')
    parser.add_argument('--output', type=str, default='eval/results.jsonl', help='Path to output JSONL file')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', help='API URL')
    parser.add_argument('--namespace', type=str, default='dev', help='Pinecone namespace')
    
    args = parser.parse_args()
    
    runner = EvalRunner(api_url=args.api_url, namespace=args.namespace)
    runner.run_evaluation(args.queries, args.output)


if __name__ == '__main__':
    main()

