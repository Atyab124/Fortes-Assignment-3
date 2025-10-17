"""Evaluation harness for the RAG system."""

import json
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from dataclasses import dataclass
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    question: str
    expected_answer: str
    predicted_answer: str
    expected_sources: List[str]
    predicted_sources: List[Dict[str, Any]]
    exact_match: bool
    f1_score: float
    similarity_score: float
    retrieval_success: bool
    grounding_score: float
    attribution_score: float
    response_time: float
    metadata: Dict[str, Any]

class MetricsCalculator:
    """Calculates various evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }
    
    def calculate_exact_match(self, predicted: str, expected: str) -> bool:
        """Calculate exact match score."""
        return predicted.strip().lower() == expected.strip().lower()
    
    def calculate_f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score between predicted and expected answers."""
        pred_tokens = self._tokenize(predicted)
        expected_tokens = self._tokenize(expected)
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # Calculate precision and recall
        pred_set = set(pred_tokens)
        expected_set = set(expected_tokens)
        
        if not pred_set:
            return 0.0
        
        true_positives = len(pred_set.intersection(expected_set))
        precision = true_positives / len(pred_set)
        recall = true_positives / len(expected_set)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_similarity_score(self, predicted: str, expected: str) -> float:
        """Calculate semantic similarity score."""
        # Simple token-based similarity (can be enhanced with embeddings)
        pred_tokens = self._tokenize(predicted)
        expected_tokens = self._tokenize(expected)
        
        if not pred_tokens and not expected_tokens:
            return 1.0
        if not pred_tokens or not expected_tokens:
            return 0.0
        
        # Jaccard similarity
        pred_set = set(pred_tokens)
        expected_set = set(expected_tokens)
        
        intersection = len(pred_set.intersection(expected_set))
        union = len(pred_set.union(expected_set))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_retrieval_success(self, predicted_sources: List[Dict[str, Any]], 
                                  expected_sources: List[str]) -> bool:
        """Check if retrieval was successful."""
        if not expected_sources:
            return True  # No expected sources
        
        predicted_filenames = {source.get('filename', '') for source in predicted_sources}
        expected_filenames = set(expected_sources)
        
        # Check if any expected source is in predicted sources
        return len(predicted_filenames.intersection(expected_filenames)) > 0
    
    def calculate_grounding_score(self, predicted_sources: List[Dict[str, Any]]) -> float:
        """Calculate grounding score based on source quality."""
        if not predicted_sources:
            return 0.0
        
        # Average similarity score of retrieved sources
        similarities = [source.get('similarity', 0.0) for source in predicted_sources]
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def calculate_attribution_score(self, attribution_report: Dict[str, Any]) -> float:
        """Calculate attribution score from attribution report."""
        if not attribution_report:
            return 0.0
        
        return attribution_report.get('quality_score', 0.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

class RAGEvaluator:
    """Main RAG evaluation class."""
    
    def __init__(self, rag_system, guardrails=None, attribution_analyzer=None):
        """Initialize RAG evaluator."""
        self.rag_system = rag_system
        self.guardrails = guardrails
        self.attribution_analyzer = attribution_analyzer
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_question(self, question: str, expected_answer: str, 
                         expected_sources: List[str] = None) -> EvaluationResult:
        """Evaluate a single question."""
        start_time = time.time()
        
        # Validate query through guardrails if available
        if self.guardrails:
            is_valid, processed_query, validation_metadata = self.guardrails.validate_query(question)
            if not is_valid:
                return EvaluationResult(
                    question=question,
                    expected_answer=expected_answer,
                    predicted_answer=f"Query rejected: {processed_query}",
                    expected_sources=expected_sources or [],
                    predicted_sources=[],
                    exact_match=False,
                    f1_score=0.0,
                    similarity_score=0.0,
                    retrieval_success=False,
                    grounding_score=0.0,
                    attribution_score=0.0,
                    response_time=time.time() - start_time,
                    metadata={'validation_error': validation_metadata}
                )
            question = processed_query
        
        # Query the RAG system
        try:
            result = self.rag_system.query(question)
            predicted_answer = result.get('answer', '')
            predicted_sources = result.get('sources', [])
            retrieval_success = result.get('retrieval_success', False)
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            predicted_answer = f"Error: {str(e)}"
            predicted_sources = []
            retrieval_success = False
        
        response_time = time.time() - start_time
        
        # Calculate metrics
        exact_match = self.metrics_calculator.calculate_exact_match(predicted_answer, expected_answer)
        f1_score = self.metrics_calculator.calculate_f1_score(predicted_answer, expected_answer)
        similarity_score = self.metrics_calculator.calculate_similarity_score(predicted_answer, expected_answer)
        retrieval_success_metric = self.metrics_calculator.calculate_retrieval_success(
            predicted_sources, expected_sources or []
        )
        grounding_score = self.metrics_calculator.calculate_grounding_score(predicted_sources)
        
        # Calculate attribution score if analyzer is available
        attribution_score = 0.0
        attribution_metadata = {}
        if self.attribution_analyzer and predicted_sources:
            try:
                attributions = self.attribution_analyzer.analyze_response(predicted_answer, predicted_sources)
                attribution_report = self.attribution_analyzer.generate_attribution_report(attributions)
                attribution_score = attribution_report.get('quality_score', 0.0)
                attribution_metadata = attribution_report
            except Exception as e:
                logger.error(f"Error in attribution analysis: {e}")
                attribution_metadata = {'error': str(e)}
        
        metadata = {
            'rag_result': result,
            'attribution_metadata': attribution_metadata
        }
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            predicted_answer=predicted_answer,
            expected_sources=expected_sources or [],
            predicted_sources=predicted_sources,
            exact_match=exact_match,
            f1_score=f1_score,
            similarity_score=similarity_score,
            retrieval_success=retrieval_success_metric,
            grounding_score=grounding_score,
            attribution_score=attribution_score,
            response_time=response_time,
            metadata=metadata
        )
    
    def evaluate_dataset(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a dataset of questions."""
        results = []
        
        for i, item in enumerate(eval_data):
            logger.info(f"Evaluating question {i+1}/{len(eval_data)}: {item['question'][:50]}...")
            
            result = self.evaluate_question(
                question=item['question'],
                expected_answer=item.get('expected_answer', ''),
                expected_sources=item.get('expected_sources', [])
            )
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics,
            'total_questions': len(results),
            'evaluation_timestamp': time.time()
        }
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        # Basic metrics
        total_questions = len(results)
        exact_matches = sum(1 for r in results if r.exact_match)
        successful_retrievals = sum(1 for r in results if r.retrieval_success)
        
        # Average scores
        avg_f1 = sum(r.f1_score for r in results) / total_questions
        avg_similarity = sum(r.similarity_score for r in results) / total_questions
        avg_grounding = sum(r.grounding_score for r in results) / total_questions
        avg_attribution = sum(r.attribution_score for r in results) / total_questions
        avg_response_time = sum(r.response_time for r in results) / total_questions
        
        # Performance by question type (if available)
        performance_by_type = {}
        
        # Error analysis
        errors = [r for r in results if 'validation_error' in r.metadata]
        retrieval_failures = [r for r in results if not r.retrieval_success]
        
        return {
            'exact_match_rate': exact_matches / total_questions,
            'f1_score': avg_f1,
            'similarity_score': avg_similarity,
            'retrieval_success_rate': successful_retrievals / total_questions,
            'grounding_score': avg_grounding,
            'attribution_score': avg_attribution,
            'average_response_time': avg_response_time,
            'total_questions': total_questions,
            'successful_retrievals': successful_retrievals,
            'exact_matches': exact_matches,
            'errors': len(errors),
            'retrieval_failures': len(retrieval_failures),
            'performance_by_type': performance_by_type
        }

class EvaluationLoader:
    """Loads evaluation datasets from YAML files."""
    
    @staticmethod
    def load_eval_config(config_path: str) -> List[Dict[str, Any]]:
        """Load evaluation configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('questions', [])
    
    @staticmethod
    def save_eval_report(report: Dict[str, Any], output_path: str):
        """Save evaluation report to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation report saved to {output_path}")

def run_evaluation(rag_system, eval_config_path: str = "eval.yaml", 
                  output_path: str = "eval_report.json", 
                  guardrails=None, attribution_analyzer=None):
    """Run evaluation and generate report."""
    logger.info("Starting RAG evaluation...")
    
    # Load evaluation data
    eval_data = EvaluationLoader.load_eval_config(eval_config_path)
    logger.info(f"Loaded {len(eval_data)} evaluation questions")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_system, guardrails, attribution_analyzer)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(eval_data)
    
    # Save report
    EvaluationLoader.save_eval_report(results, output_path)
    
    # Print summary
    metrics = results['aggregate_metrics']
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Similarity Score: {metrics['similarity_score']:.3f}")
    print(f"Retrieval Success Rate: {metrics['retrieval_success_rate']:.3f}")
    print(f"Grounding Score: {metrics['grounding_score']:.3f}")
    print(f"Attribution Score: {metrics['attribution_score']:.3f}")
    print(f"Average Response Time: {metrics['average_response_time']:.2f}s")
    print("="*50)
    
    return results
