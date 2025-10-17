"""Evaluation harness for the RAG system."""

import yaml
import json
import time
from typing import List, Dict, Any, Tuple
from sklearn.metrics import f1_score
import re
from difflib import SequenceMatcher

from rag_core import RAGCore, RAGResponse
from config import Config

class RAGEvaluator:
    """Evaluation harness for RAG system."""
    
    def __init__(self, rag_core: RAGCore):
        self.rag_core = rag_core
        self.eval_data = []
        self.results = []
    
    def load_eval_data(self, eval_file: str = None) -> List[Dict[str, Any]]:
        """Load evaluation data from YAML file."""
        eval_file = eval_file or Config.EVAL_FILE
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                self.eval_data = yaml.safe_load(f)
            return self.eval_data
        except FileNotFoundError:
            print(f"Evaluation file {eval_file} not found. Creating sample data.")
            self._create_sample_eval_data()
            return self.eval_data
        except Exception as e:
            print(f"Error loading evaluation data: {e}")
            return []
    
    def _create_sample_eval_data(self):
        """Create sample evaluation data."""
        self.eval_data = [
            {
                "id": 1,
                "question": "What is the main topic of the document?",
                "expected_answer": "The document discusses machine learning applications.",
                "expected_citations": ["doc1.md#1", "doc1.md#2"],
                "category": "factual"
            },
            {
                "id": 2,
                "question": "How does the system work?",
                "expected_answer": "The system uses a neural network approach.",
                "expected_citations": ["doc2.md#3", "doc2.md#4"],
                "category": "procedural"
            },
            {
                "id": 3,
                "question": "What are the benefits?",
                "expected_answer": "The benefits include improved accuracy and efficiency.",
                "expected_citations": ["doc3.md#1", "doc3.md#2"],
                "category": "analytical"
            }
        ]
        
        # Save sample data
        with open(Config.EVAL_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self.eval_data, f, default_flow_style=False)
    
    def run_evaluation(self, eval_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run evaluation on the RAG system."""
        if eval_data is None:
            eval_data = self.eval_data
        
        if not eval_data:
            return {"error": "No evaluation data available"}
        
        results = []
        total_questions = len(eval_data)
        
        print(f"Running evaluation on {total_questions} questions...")
        
        for i, item in enumerate(eval_data):
            print(f"Processing question {i+1}/{total_questions}: {item['question'][:50]}...")
            
            # Query the RAG system
            start_time = time.time()
            response = self.rag_core.query(item["question"])
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                item["question"],
                item["expected_answer"],
                item.get("expected_citations", []),
                response
            )
            
            result = {
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "expected_citations": item.get("expected_citations", []),
                "actual_answer": response.answer,
                "actual_citations": response.citations,
                "metrics": metrics,
                "processing_time": processing_time,
                "category": item.get("category", "unknown")
            }
            
            results.append(result)
        
        self.results = results
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)
        
        return {
            "overall_metrics": overall_metrics,
            "detailed_results": results,
            "total_questions": total_questions
        }
    
    def _calculate_metrics(self, question: str, expected_answer: str, 
                          expected_citations: List[str], response: RAGResponse) -> Dict[str, Any]:
        """Calculate metrics for a single question."""
        # Exact Match (EM)
        em_score = 1.0 if expected_answer.lower().strip() == response.answer.lower().strip() else 0.0
        
        # F1 Score
        f1_score = self._calculate_f1_score(expected_answer, response.answer)
        
        # Similarity Score
        similarity_score = self._calculate_similarity(expected_answer, response.answer)
        
        # Citation Accuracy
        citation_accuracy = self._calculate_citation_accuracy(expected_citations, response.citations)
        
        # Hallucination Detection
        hallucination_detected = response.hallucination_detected
        
        # Grounding Score
        grounding_score = response.grounding_score
        
        return {
            "exact_match": em_score,
            "f1_score": f1_score,
            "similarity_score": similarity_score,
            "citation_accuracy": citation_accuracy,
            "hallucination_detected": hallucination_detected,
            "grounding_score": grounding_score,
            "tokens_used": response.tokens_used,
            "estimated_cost": response.estimated_cost
        }
    
    def _calculate_f1_score(self, expected: str, actual: str) -> float:
        """Calculate F1 score between expected and actual answers."""
        expected_tokens = set(expected.lower().split())
        actual_tokens = set(actual.lower().split())
        
        if not expected_tokens and not actual_tokens:
            return 1.0
        
        if not expected_tokens or not actual_tokens:
            return 0.0
        
        # Calculate precision and recall
        true_positives = len(expected_tokens.intersection(actual_tokens))
        precision = true_positives / len(actual_tokens) if actual_tokens else 0.0
        recall = true_positives / len(expected_tokens) if expected_tokens else 0.0
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity score between expected and actual answers."""
        return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
    
    def _calculate_citation_accuracy(self, expected_citations: List[str], 
                                   actual_citations: List[Dict[str, Any]]) -> float:
        """Calculate citation accuracy."""
        if not expected_citations:
            return 1.0 if not actual_citations else 0.0
        
        if not actual_citations:
            return 0.0
        
        # Extract document names from expected citations
        expected_docs = set()
        for citation in expected_citations:
            if '#' in citation:
                doc_name = citation.split('#')[0]
                expected_docs.add(doc_name)
            else:
                expected_docs.add(citation)
        
        # Extract document names from actual citations
        actual_docs = set()
        for citation in actual_citations:
            filename = citation.get("filename", "")
            if filename:
                actual_docs.add(filename)
        
        # Calculate accuracy
        if not expected_docs:
            return 1.0
        
        intersection = len(expected_docs.intersection(actual_docs))
        union = len(expected_docs.union(actual_docs))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall metrics across all questions."""
        if not results:
            return {}
        
        # Aggregate metrics
        total_questions = len(results)
        exact_matches = sum(1 for r in results if r["metrics"]["exact_match"] == 1.0)
        avg_f1 = sum(r["metrics"]["f1_score"] for r in results) / total_questions
        avg_similarity = sum(r["metrics"]["similarity_score"] for r in results) / total_questions
        avg_citation_accuracy = sum(r["metrics"]["citation_accuracy"] for r in results) / total_questions
        avg_grounding_score = sum(r["metrics"]["grounding_score"] for r in results) / total_questions
        
        # Hallucination statistics
        hallucination_count = sum(1 for r in results if r["metrics"]["hallucination_detected"])
        hallucination_rate = hallucination_count / total_questions
        
        # Cost and performance
        total_tokens = sum(r["metrics"]["tokens_used"] for r in results)
        total_cost = sum(r["metrics"]["estimated_cost"] for r in results)
        avg_processing_time = sum(r["processing_time"] for r in results) / total_questions
        
        # Category-wise metrics
        category_metrics = {}
        for result in results:
            category = result["category"]
            if category not in category_metrics:
                category_metrics[category] = {
                    "count": 0,
                    "exact_matches": 0,
                    "f1_scores": [],
                    "similarity_scores": []
                }
            
            category_metrics[category]["count"] += 1
            if result["metrics"]["exact_match"] == 1.0:
                category_metrics[category]["exact_matches"] += 1
            category_metrics[category]["f1_scores"].append(result["metrics"]["f1_score"])
            category_metrics[category]["similarity_scores"].append(result["metrics"]["similarity_score"])
        
        # Calculate category averages
        for category in category_metrics:
            metrics = category_metrics[category]
            metrics["exact_match_rate"] = metrics["exact_matches"] / metrics["count"]
            metrics["avg_f1"] = sum(metrics["f1_scores"]) / len(metrics["f1_scores"])
            metrics["avg_similarity"] = sum(metrics["similarity_scores"]) / len(metrics["similarity_scores"])
        
        return {
            "total_questions": total_questions,
            "exact_match_rate": exact_matches / total_questions,
            "average_f1_score": avg_f1,
            "average_similarity_score": avg_similarity,
            "average_citation_accuracy": avg_citation_accuracy,
            "average_grounding_score": avg_grounding_score,
            "hallucination_rate": hallucination_rate,
            "total_tokens_used": total_tokens,
            "total_estimated_cost": total_cost,
            "average_processing_time": avg_processing_time,
            "category_metrics": category_metrics
        }
    
    def save_results(self, output_file: str = None) -> str:
        """Save evaluation results to JSON file."""
        output_file = output_file or Config.EVAL_OUTPUT
        
        results_data = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_metrics": self._calculate_overall_metrics(self.results),
            "detailed_results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No evaluation results available.")
            return
        
        overall_metrics = self._calculate_overall_metrics(self.results)
        
        print("\n" + "="*50)
        print("RAG EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions: {overall_metrics['total_questions']}")
        print(f"Exact Match Rate: {overall_metrics['exact_match_rate']:.3f}")
        print(f"Average F1 Score: {overall_metrics['average_f1_score']:.3f}")
        print(f"Average Similarity: {overall_metrics['average_similarity_score']:.3f}")
        print(f"Citation Accuracy: {overall_metrics['average_citation_accuracy']:.3f}")
        print(f"Grounding Score: {overall_metrics['average_grounding_score']:.3f}")
        print(f"Hallucination Rate: {overall_metrics['hallucination_rate']:.3f}")
        print(f"Total Tokens: {overall_metrics['total_tokens_used']}")
        print(f"Estimated Cost: ${overall_metrics['total_estimated_cost']:.4f}")
        print(f"Avg Processing Time: {overall_metrics['average_processing_time']:.3f}s")
        
        # Category breakdown
        if overall_metrics.get('category_metrics'):
            print("\nCategory Breakdown:")
            for category, metrics in overall_metrics['category_metrics'].items():
                print(f"  {category}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Exact Match: {metrics['exact_match_rate']:.3f}")
                print(f"    F1 Score: {metrics['avg_f1']:.3f}")
                print(f"    Similarity: {metrics['avg_similarity']:.3f}")
        
        print("="*50)
