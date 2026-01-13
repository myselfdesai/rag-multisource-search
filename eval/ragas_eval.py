import json
import logging
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
QUERIES_FILE = "eval/queries.json"
RESULTS_DIR = "eval/results"


def load_queries(filepath: str) -> List[Dict[str, str]]:
    """Load test queries from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {filepath}")
    return queries


def query_rag_api(question: str, namespace: str = "prod") -> Dict[str, Any]:
    """Query the RAG API and return results."""
    payload = {
        "question": question,
        "namespace": namespace,
        "top_k": 20,
        "retrieval_only": False
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying API: {e}")
        raise


def prepare_ragas_dataset(queries: List[Dict[str, str]]) -> tuple[Dataset, List[Dict[str, Any]]]:
    """
    Query RAG API for each query and prepare RAGAS dataset.
    
    Returns:
        Tuple of (RAGAS Dataset, detailed query results)
    """
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    detailed_results = []
    
    for idx, query in enumerate(queries, 1):
        question = query["question"]
        ground_truth = query.get("ground_truth", "")
        
        logger.info(f"Processing query {idx}/{len(queries)}: {question[:50]}...")
        
        try:
            # Query RAG API
            response = query_rag_api(question)
            
            answer = response.get("answer", "")
            
            # Extract context from retrieved chunks
            contexts = []
            retrieved = response.get("retrieved", [])
            for chunk in retrieved[:5]:  # Top 5 chunks as context
                contexts.append(chunk.get("text", ""))
            
            # Add to RAGAS dataset
            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append(contexts)
            ragas_data["ground_truth"].append(ground_truth)
            
            # Store detailed result
            detailed_results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "sources": response.get("sources", []),
                "retrieved_count": len(retrieved)
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Add placeholder data to maintain alignment
            ragas_data["question"].append(question)
            ragas_data["answer"].append(f"ERROR: {str(e)}")
            ragas_data["contexts"].append([])
            ragas_data["ground_truth"].append(ground_truth)
            
            detailed_results.append({
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "error": str(e)
            })
    
    # Create RAGAS dataset
    dataset = Dataset.from_dict(ragas_data)
    
    return dataset, detailed_results


def run_ragas_evaluation(dataset: Dataset) -> Dict[str, float]:
    """Run RAGAS evaluation and return metrics."""
    logger.info("Running RAGAS evaluation...")
    
    # Define metrics to evaluate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    
    # Run evaluation
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Initialize models explicitly to avoid async/sync conflicts
    # Use cheaper/faster models for evaluation if possible
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    results = evaluate(
        dataset, 
        metrics=metrics,
        llm=llm, 
        embeddings=embeddings
    )
    
    # Extract scores from EvaluationResult object
    # Convert to dictionary if it's not already
    if hasattr(results, 'to_pandas'):
        # It's an EvaluationResult object, convert to DataFrame first
        results_df = results.to_pandas()
        scores = {}
        # Only compute mean for numeric columns (the actual metric scores)
        # Skip metadata columns like 'question', 'answer', 'contexts', 'ground_truth'
        for col in results_df.columns:
            if results_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                scores[col] = float(results_df[col].mean())
    elif hasattr(results, 'keys'):
        # It's already a dict-like object
        scores = {}
        for metric_name in results.keys():
            if metric_name not in ['question', 'answer', 'contexts', 'ground_truth']:
                try:
                    scores[metric_name] = float(results[metric_name])
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass
    else:
        # Fallback - try to access as dict
        scores = dict(results)
    
    logger.info("RAGAS evaluation complete")
    return scores, results


def save_results(
    overall_metrics: Dict[str, float],
    detailed_results: List[Dict[str, Any]],
    per_query_metrics: pd.DataFrame
):
    """Save evaluation results to JSON and CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine detailed results with per-query metrics
    for i, result in enumerate(detailed_results):
        if i < len(per_query_metrics):
            result["metrics"] = per_query_metrics.iloc[i].to_dict()
    
    # Prepare full results
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": overall_metrics,
        "query_count": len(detailed_results),
        "queries": detailed_results
    }
    
    # Save detailed JSON
    json_path = Path(RESULTS_DIR) / f"detailed_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed results to {json_path}")
    
    # Also save as latest
    latest_json = Path(RESULTS_DIR) / "latest_results.json"
    with open(latest_json, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved latest results to {latest_json}")
    
    # Save summary CSV
    summary_df = pd.DataFrame([overall_metrics])
    csv_path = Path(RESULTS_DIR) / f"summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")
    
    # Save per-query metrics CSV
    per_query_csv = Path(RESULTS_DIR) / f"per_query_{timestamp}.csv"
    per_query_metrics.to_csv(per_query_csv, index=False)
    logger.info(f"Saved per-query metrics to {per_query_csv}")


def main():
    """Main evaluation workflow."""
    logger.info("="*60)
    logger.info("Starting RAGAS Evaluation")
    logger.info("="*60)
    
    # Check API availability
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"API is not healthy at {API_URL}")
            return
    except requests.exceptions.RequestException:
        logger.error(f"Cannot connect to API at {API_URL}")
        logger.error("Please start the FastAPI server: python -m app.main")
        return
    
    # Load queries
    queries = load_queries(QUERIES_FILE)
    if not queries:
        logger.error("No queries found")
        return
    
    # Prepare dataset
    logger.info("Querying RAG API for all test queries...")
    dataset, detailed_results = prepare_ragas_dataset(queries)
    
    # Run evaluation
    overall_metrics, results = run_ragas_evaluation(dataset)
    
    # Convert to DataFrame for per-query metrics
    if hasattr(results, 'to_pandas'):
        per_query_df = results.to_pandas()
    else:
        per_query_df = pd.DataFrame(results)
    
    # Display results
    logger.info("="*60)
    logger.info("OVERALL METRICS")
    logger.info("="*60)
    for metric, score in overall_metrics.items():
        logger.info(f"{metric:25s}: {score:.4f}")
    
    # Save results
    save_results(overall_metrics, detailed_results, per_query_df)
    
    logger.info("="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
