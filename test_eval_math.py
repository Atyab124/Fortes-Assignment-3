"""Tests for evaluation math and metrics."""

import unittest
import tempfile
import os
import yaml

from evaluation import RAGEvaluator
from rag_core import RAGCore, RAGResponse

class TestEvalMath(unittest.TestCase):
    """Test cases for evaluation math and metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock RAG core
        self.rag_core = RAGCore(use_local_model=True)
        
        # Create evaluator
        self.evaluator = RAGEvaluator(self.rag_core)
        
        # Create sample eval data
        self.sample_eval_data = [
            {
                "id": 1,
                "question": "What is machine learning?",
                "expected_answer": "Machine learning is a subset of artificial intelligence.",
                "expected_citations": ["doc1.md#1"],
                "category": "factual"
            },
            {
                "id": 2,
                "question": "How does neural networks work?",
                "expected_answer": "Neural networks are inspired by biological neural networks.",
                "expected_citations": ["doc2.md#2"],
                "category": "technical"
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_calculate_f1_score_perfect_match(self):
        """Test F1 score calculation with perfect match."""
        expected = "Machine learning is a subset of AI"
        actual = "Machine learning is a subset of AI"
        
        f1 = self.evaluator._calculate_f1_score(expected, actual)
        self.assertEqual(f1, 1.0)
    
    def test_calculate_f1_score_no_match(self):
        """Test F1 score calculation with no match."""
        expected = "Machine learning is a subset of AI"
        actual = "Deep learning is a subset of AI"
        
        f1 = self.evaluator._calculate_f1_score(expected, actual)
        self.assertGreater(f1, 0.0)  # Should have some overlap
        self.assertLess(f1, 1.0)
    
    def test_calculate_f1_score_empty_strings(self):
        """Test F1 score calculation with empty strings."""
        f1 = self.evaluator._calculate_f1_score("", "")
        self.assertEqual(f1, 1.0)
        
        f1 = self.evaluator._calculate_f1_score("", "some text")
        self.assertEqual(f1, 0.0)
        
        f1 = self.evaluator._calculate_f1_score("some text", "")
        self.assertEqual(f1, 0.0)
    
    def test_calculate_f1_score_partial_match(self):
        """Test F1 score calculation with partial match."""
        expected = "Machine learning is a subset of artificial intelligence"
        actual = "Machine learning is a subset of AI"
        
        f1 = self.evaluator._calculate_f1_score(expected, actual)
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)
    
    def test_calculate_similarity_perfect_match(self):
        """Test similarity calculation with perfect match."""
        expected = "Machine learning is a subset of AI"
        actual = "Machine learning is a subset of AI"
        
        similarity = self.evaluator._calculate_similarity(expected, actual)
        self.assertEqual(similarity, 1.0)
    
    def test_calculate_similarity_no_match(self):
        """Test similarity calculation with no match."""
        expected = "Machine learning is a subset of AI"
        actual = "Deep learning is a subset of AI"
        
        similarity = self.evaluator._calculate_similarity(expected, actual)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_calculate_similarity_case_insensitive(self):
        """Test similarity calculation is case insensitive."""
        expected = "Machine learning is a subset of AI"
        actual = "MACHINE LEARNING IS A SUBSET OF AI"
        
        similarity = self.evaluator._calculate_similarity(expected, actual)
        self.assertAlmostEqual(similarity, 1.0, places=2)
    
    def test_calculate_citation_accuracy_perfect_match(self):
        """Test citation accuracy with perfect match."""
        expected_citations = ["doc1.md#1", "doc2.md#2"]
        actual_citations = [
            {"filename": "doc1.md", "chunk_id": "1"},
            {"filename": "doc2.md", "chunk_id": "2"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 1.0)
    
    def test_calculate_citation_accuracy_no_match(self):
        """Test citation accuracy with no match."""
        expected_citations = ["doc1.md#1", "doc2.md#2"]
        actual_citations = [
            {"filename": "doc3.md", "chunk_id": "1"},
            {"filename": "doc4.md", "chunk_id": "2"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 0.0)
    
    def test_calculate_citation_accuracy_partial_match(self):
        """Test citation accuracy with partial match."""
        expected_citations = ["doc1.md#1", "doc2.md#2"]
        actual_citations = [
            {"filename": "doc1.md", "chunk_id": "1"},
            {"filename": "doc3.md", "chunk_id": "2"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertGreater(accuracy, 0.0)
        self.assertLess(accuracy, 1.0)
    
    def test_calculate_citation_accuracy_empty_expected(self):
        """Test citation accuracy with empty expected citations."""
        expected_citations = []
        actual_citations = [
            {"filename": "doc1.md", "chunk_id": "1"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 1.0)  # Should be 1.0 when no expected citations
    
    def test_calculate_citation_accuracy_empty_actual(self):
        """Test citation accuracy with empty actual citations."""
        expected_citations = ["doc1.md#1"]
        actual_citations = []
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 0.0)
    
    def test_calculate_citation_accuracy_with_line_numbers(self):
        """Test citation accuracy with line numbers in expected citations."""
        expected_citations = ["doc1.md#1", "doc2.md#2"]
        actual_citations = [
            {"filename": "doc1.md", "chunk_id": "1"},
            {"filename": "doc2.md", "chunk_id": "2"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 1.0)
    
    def test_calculate_citation_accuracy_without_line_numbers(self):
        """Test citation accuracy without line numbers in expected citations."""
        expected_citations = ["doc1.md", "doc2.md"]
        actual_citations = [
            {"filename": "doc1.md", "chunk_id": "1"},
            {"filename": "doc2.md", "chunk_id": "2"}
        ]
        
        accuracy = self.evaluator._calculate_citation_accuracy(expected_citations, actual_citations)
        self.assertEqual(accuracy, 1.0)
    
    def test_calculate_metrics_comprehensive(self):
        """Test comprehensive metrics calculation."""
        question = "What is machine learning?"
        expected_answer = "Machine learning is a subset of artificial intelligence."
        expected_citations = ["doc1.md#1"]
        
        # Create mock response
        response = RAGResponse(
            answer="Machine learning is a subset of artificial intelligence.",
            citations=[{"filename": "doc1.md", "chunk_id": "1", "content": "ML is a subset of AI"}],
            hallucination_detected=False,
            grounding_score=0.9,
            tokens_used=50,
            estimated_cost=0.001,
            processing_time=1.0
        )
        
        metrics = self.evaluator._calculate_metrics(
            question, expected_answer, expected_citations, response
        )
        
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["similarity_score"], 1.0)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertFalse(metrics["hallucination_detected"])
        self.assertEqual(metrics["grounding_score"], 0.9)
        self.assertEqual(metrics["tokens_used"], 50)
        self.assertEqual(metrics["estimated_cost"], 0.001)
    
    def test_calculate_overall_metrics(self):
        """Test overall metrics calculation."""
        # Create mock results
        results = [
            {
                "metrics": {
                    "exact_match": 1.0,
                    "f1_score": 1.0,
                    "similarity_score": 1.0,
                    "citation_accuracy": 1.0,
                    "hallucination_detected": False,
                    "grounding_score": 0.9,
                    "tokens_used": 50,
                    "estimated_cost": 0.001
                },
                "processing_time": 1.0,
                "category": "factual"
            },
            {
                "metrics": {
                    "exact_match": 0.0,
                    "f1_score": 0.8,
                    "similarity_score": 0.8,
                    "citation_accuracy": 0.5,
                    "hallucination_detected": True,
                    "grounding_score": 0.7,
                    "tokens_used": 40,
                    "estimated_cost": 0.0008
                },
                "processing_time": 1.5,
                "category": "technical"
            }
        ]
        
        overall_metrics = self.evaluator._calculate_overall_metrics(results)
        
        self.assertEqual(overall_metrics["total_questions"], 2)
        self.assertEqual(overall_metrics["exact_match_rate"], 0.5)
        self.assertEqual(overall_metrics["average_f1_score"], 0.9)
        self.assertEqual(overall_metrics["average_similarity_score"], 0.9)
        self.assertEqual(overall_metrics["average_citation_accuracy"], 0.75)
        self.assertEqual(overall_metrics["average_grounding_score"], 0.8)
        self.assertEqual(overall_metrics["hallucination_rate"], 0.5)
        self.assertEqual(overall_metrics["total_tokens_used"], 90)
        self.assertEqual(overall_metrics["total_estimated_cost"], 0.0018)
        self.assertEqual(overall_metrics["average_processing_time"], 1.25)
        
        # Check category breakdown
        self.assertIn("category_metrics", overall_metrics)
        self.assertIn("factual", overall_metrics["category_metrics"])
        self.assertIn("technical", overall_metrics["category_metrics"])
    
    def test_calculate_overall_metrics_empty(self):
        """Test overall metrics calculation with empty results."""
        overall_metrics = self.evaluator._calculate_overall_metrics([])
        self.assertEqual(overall_metrics, {})
    
    def test_load_eval_data(self):
        """Test loading evaluation data."""
        # Create test eval file
        eval_file = os.path.join(self.temp_dir, "test_eval.yaml")
        with open(eval_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.sample_eval_data, f, default_flow_style=False)
        
        # Load data
        loaded_data = self.evaluator.load_eval_data(eval_file)
        
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0]["id"], 1)
        self.assertEqual(loaded_data[0]["question"], "What is machine learning?")
    
    def test_load_eval_data_nonexistent(self):
        """Test loading non-existent eval file."""
        # Should create sample data
        eval_data = self.evaluator.load_eval_data("nonexistent.yaml")
        self.assertGreater(len(eval_data), 0)
    
    def test_create_sample_eval_data(self):
        """Test creation of sample evaluation data."""
        self.evaluator._create_sample_eval_data()
        
        self.assertGreater(len(self.evaluator.eval_data), 0)
        self.assertTrue(all("id" in item for item in self.evaluator.eval_data))
        self.assertTrue(all("question" in item for item in self.evaluator.eval_data))
        self.assertTrue(all("expected_answer" in item for item in self.evaluator.eval_data))
    
    def test_save_results(self):
        """Test saving evaluation results."""
        # Create mock results
        self.evaluator.results = [
            {
                "id": 1,
                "question": "What is ML?",
                "expected_answer": "Machine learning",
                "actual_answer": "Machine learning",
                "metrics": {
                    "exact_match": 1.0,
                    "f1_score": 1.0,
                    "similarity_score": 1.0,
                    "citation_accuracy": 1.0,
                    "hallucination_detected": False,
                    "grounding_score": 0.9,
                    "tokens_used": 50,
                    "estimated_cost": 0.001
                },
                "processing_time": 1.0,
                "category": "factual"
            }
        ]
        
        # Save results
        output_file = os.path.join(self.temp_dir, "test_results.json")
        saved_file = self.evaluator.save_results(output_file)
        
        self.assertEqual(saved_file, output_file)
        self.assertTrue(os.path.exists(output_file))
        
        # Verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
        
        self.assertIn("evaluation_timestamp", data)
        self.assertIn("overall_metrics", data)
        self.assertIn("detailed_results", data)
        self.assertEqual(len(data["detailed_results"]), 1)
    
    def test_f1_score_edge_cases(self):
        """Test F1 score edge cases."""
        # Test with very similar but not identical text
        expected = "The quick brown fox jumps over the lazy dog"
        actual = "The quick brown fox jumps over the lazy dog."
        
        f1 = self.evaluator._calculate_f1_score(expected, actual)
        self.assertGreater(f1, 0.9)  # Should be very high
        
        # Test with completely different text
        expected = "Machine learning is great"
        actual = "Deep learning is awesome"
        
        f1 = self.evaluator._calculate_f1_score(expected, actual)
        self.assertGreater(f1, 0.0)  # Should have some overlap
        self.assertLess(f1, 1.0)
    
    def test_similarity_edge_cases(self):
        """Test similarity edge cases."""
        # Test with identical text
        expected = "Machine learning is a subset of AI"
        actual = "Machine learning is a subset of AI"
        
        similarity = self.evaluator._calculate_similarity(expected, actual)
        self.assertEqual(similarity, 1.0)
        
        # Test with completely different text
        expected = "Machine learning is a subset of AI"
        actual = "Deep learning is a subset of AI"
        
        similarity = self.evaluator._calculate_similarity(expected, actual)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_citation_accuracy_edge_cases(self):
        """Test citation accuracy edge cases."""
        # Test with no citations
        accuracy = self.evaluator._calculate_citation_accuracy([], [])
        self.assertEqual(accuracy, 1.0)
        
        # Test with empty expected but actual citations
        accuracy = self.evaluator._calculate_citation_accuracy([], [{"filename": "doc1.md"}])
        self.assertEqual(accuracy, 1.0)
        
        # Test with empty actual but expected citations
        accuracy = self.evaluator._calculate_citation_accuracy(["doc1.md"], [])
        self.assertEqual(accuracy, 0.0)

if __name__ == '__main__':
    unittest.main()
