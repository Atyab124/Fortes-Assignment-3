"""Tests for evaluation metrics and math calculations."""

import pytest
import numpy as np

from evaluation import MetricsCalculator, EvaluationResult
from attribution import AttributionAnalyzer, AttributionLevel

class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
    
    def test_exact_match_identical(self):
        """Test exact match for identical strings."""
        predicted = "Machine learning is a subset of artificial intelligence."
        expected = "Machine learning is a subset of artificial intelligence."
        
        result = self.calculator.calculate_exact_match(predicted, expected)
        assert result is True
    
    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        predicted = "Machine learning is a subset of artificial intelligence."
        expected = "MACHINE LEARNING IS A SUBSET OF ARTIFICIAL INTELLIGENCE."
        
        result = self.calculator.calculate_exact_match(predicted, expected)
        assert result is True
    
    def test_exact_match_whitespace(self):
        """Test exact match handles whitespace differences."""
        predicted = "Machine learning is a subset of artificial intelligence."
        expected = "  Machine learning is a subset of artificial intelligence.  "
        
        result = self.calculator.calculate_exact_match(predicted, expected)
        assert result is True
    
    def test_exact_match_different(self):
        """Test exact match for different strings."""
        predicted = "Machine learning is a subset of AI."
        expected = "Machine learning is a subset of artificial intelligence."
        
        result = self.calculator.calculate_exact_match(predicted, expected)
        assert result is False
    
    def test_f1_score_perfect_match(self):
        """Test F1 score for perfect match."""
        predicted = "Machine learning is artificial intelligence"
        expected = "Machine learning is artificial intelligence"
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert abs(f1 - 1.0) < 0.01
    
    def test_f1_score_partial_match(self):
        """Test F1 score for partial match."""
        predicted = "Machine learning is a subset of AI"
        expected = "Machine learning is a subset of artificial intelligence"
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert 0.5 < f1 < 1.0
    
    def test_f1_score_no_match(self):
        """Test F1 score for no match."""
        predicted = "Deep learning uses neural networks"
        expected = "Machine learning is a subset of artificial intelligence"
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert f1 < 0.5
    
    def test_f1_score_empty_predicted(self):
        """Test F1 score with empty predicted text."""
        predicted = ""
        expected = "Machine learning is a subset of artificial intelligence"
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert f1 == 0.0
    
    def test_f1_score_empty_expected(self):
        """Test F1 score with empty expected text."""
        predicted = "Machine learning is a subset of artificial intelligence"
        expected = ""
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert f1 == 0.0
    
    def test_f1_score_both_empty(self):
        """Test F1 score with both texts empty."""
        predicted = ""
        expected = ""
        
        f1 = self.calculator.calculate_f1_score(predicted, expected)
        assert f1 == 1.0
    
    def test_similarity_score_identical(self):
        """Test similarity score for identical texts."""
        predicted = "Machine learning is artificial intelligence"
        expected = "Machine learning is artificial intelligence"
        
        similarity = self.calculator.calculate_similarity_score(predicted, expected)
        assert abs(similarity - 1.0) < 0.01
    
    def test_similarity_score_partial(self):
        """Test similarity score for partially similar texts."""
        predicted = "Machine learning is a subset of AI"
        expected = "Machine learning is a subset of artificial intelligence"
        
        similarity = self.calculator.calculate_similarity_score(predicted, expected)
        assert 0.3 < similarity < 1.0
    
    def test_similarity_score_different(self):
        """Test similarity score for different texts."""
        predicted = "Deep learning uses neural networks"
        expected = "Machine learning is a subset of artificial intelligence"
        
        similarity = self.calculator.calculate_similarity_score(predicted, expected)
        assert similarity < 0.5
    
    def test_retrieval_success_with_sources(self):
        """Test retrieval success when sources match."""
        predicted_sources = [
            {'filename': 'ai_basics.md'},
            {'filename': 'ml_guide.txt'}
        ]
        expected_sources = ['ai_basics.md']
        
        success = self.calculator.calculate_retrieval_success(predicted_sources, expected_sources)
        assert success is True
    
    def test_retrieval_success_no_match(self):
        """Test retrieval success when sources don't match."""
        predicted_sources = [
            {'filename': 'deep_learning.md'},
            {'filename': 'neural_networks.txt'}
        ]
        expected_sources = ['ai_basics.md']
        
        success = self.calculator.calculate_retrieval_success(predicted_sources, expected_sources)
        assert success is False
    
    def test_retrieval_success_no_expected(self):
        """Test retrieval success when no expected sources."""
        predicted_sources = [
            {'filename': 'ai_basics.md'}
        ]
        expected_sources = []
        
        success = self.calculator.calculate_retrieval_success(predicted_sources, expected_sources)
        assert success is True
    
    def test_retrieval_success_no_predicted(self):
        """Test retrieval success when no predicted sources."""
        predicted_sources = []
        expected_sources = ['ai_basics.md']
        
        success = self.calculator.calculate_retrieval_success(predicted_sources, expected_sources)
        assert success is False
    
    def test_grounding_score(self):
        """Test grounding score calculation."""
        predicted_sources = [
            {'similarity': 0.9},
            {'similarity': 0.8},
            {'similarity': 0.7}
        ]
        
        score = self.calculator.calculate_grounding_score(predicted_sources)
        assert abs(score - 0.8) < 0.01  # Average of similarities
    
    def test_grounding_score_empty(self):
        """Test grounding score with empty sources."""
        predicted_sources = []
        
        score = self.calculator.calculate_grounding_score(predicted_sources)
        assert score == 0.0
    
    def test_attribution_score(self):
        """Test attribution score calculation."""
        attribution_report = {
            'quality_score': 0.85,
            'total_sentences': 5,
            'hallucinated_sentences': 1
        }
        
        score = self.calculator.calculate_attribution_score(attribution_report)
        assert score == 0.85
    
    def test_attribution_score_empty(self):
        """Test attribution score with empty report."""
        attribution_report = {}
        
        score = self.calculator.calculate_attribution_score(attribution_report)
        assert score == 0.0

class TestAttributionAnalyzer:
    """Test cases for AttributionAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AttributionAnalyzer()
    
    def test_analyze_well_attributed_response(self):
        """Test analysis of well-attributed response."""
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        source_chunks = [
            {
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
                'similarity': 0.9,
                'chunk_id': 1,
                'document_id': 1,
                'document': {'filename': 'ai_basics.md'}
            }
        ]
        
        attributions = self.analyzer.analyze_response(response, source_chunks)
        
        assert len(attributions) > 0
        assert attributions[0].confidence > 0.5
        assert attributions[0].attribution_level in [AttributionLevel.HIGH, AttributionLevel.MEDIUM]
        assert not attributions[0].is_hallucinated
    
    def test_analyze_hallucinated_response(self):
        """Test analysis of hallucinated response."""
        response = "Machine learning was invented by John Smith in 1995 and is used by aliens on Mars."
        source_chunks = [
            {
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity': 0.3,
                'chunk_id': 1,
                'document_id': 1,
                'document': {'filename': 'ai_basics.md'}
            }
        ]
        
        attributions = self.analyzer.analyze_response(response, source_chunks)
        
        assert len(attributions) > 0
        # Should detect hallucination due to low similarity and specific false claims
        assert attributions[0].confidence < 0.5 or attributions[0].is_hallucinated
    
    def test_analyze_response_without_sources(self):
        """Test analysis of response without sources."""
        response = "Machine learning is a subset of artificial intelligence."
        source_chunks = []
        
        attributions = self.analyzer.analyze_response(response, source_chunks)
        
        assert len(attributions) > 0
        assert attributions[0].attribution_level == AttributionLevel.NONE
        assert attributions[0].is_hallucinated
    
    def test_generate_attribution_report(self):
        """Test attribution report generation."""
        response = "Machine learning is AI. Neural networks are used in deep learning."
        source_chunks = [
            {
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity': 0.9,
                'chunk_id': 1,
                'document_id': 1,
                'document': {'filename': 'ai_basics.md'}
            },
            {
                'content': 'Neural networks are used in deep learning applications.',
                'similarity': 0.8,
                'chunk_id': 2,
                'document_id': 1,
                'document': {'filename': 'ai_basics.md'}
            }
        ]
        
        attributions = self.analyzer.analyze_response(response, source_chunks)
        report = self.analyzer.generate_attribution_report(attributions)
        
        assert 'total_sentences' in report
        assert 'hallucinated_sentences' in report
        assert 'quality_score' in report
        assert 'attribution_counts' in report
        assert 'problematic_sentences' in report
        
        assert report['total_sentences'] > 0
        assert 0 <= report['quality_score'] <= 1
    
    def test_highlight_problematic_sentences(self):
        """Test highlighting of problematic sentences."""
        response = "Machine learning is AI. This is completely made up information. Neural networks are important."
        source_chunks = [
            {
                'content': 'Machine learning is a subset of artificial intelligence.',
                'similarity': 0.9,
                'chunk_id': 1,
                'document_id': 1,
                'document': {'filename': 'ai_basics.md'}
            }
        ]
        
        attributions = self.analyzer.analyze_response(response, source_chunks)
        highlighted = self.analyzer.highlight_problematic_sentences(response, attributions)
        
        # Should contain highlighting markers
        assert "🚨" in highlighted or "⚠️" in highlighted

class TestEvaluationResult:
    """Test cases for EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test creation of EvaluationResult."""
        result = EvaluationResult(
            question="What is machine learning?",
            expected_answer="Machine learning is a subset of AI.",
            predicted_answer="Machine learning is a subset of artificial intelligence.",
            expected_sources=["ai_basics.md"],
            predicted_sources=[{'filename': 'ai_basics.md', 'similarity': 0.9}],
            exact_match=False,
            f1_score=0.8,
            similarity_score=0.85,
            retrieval_success=True,
            grounding_score=0.9,
            attribution_score=0.8,
            response_time=2.5,
            metadata={'test': True}
        )
        
        assert result.question == "What is machine learning?"
        assert result.exact_match is False
        assert result.f1_score == 0.8
        assert result.retrieval_success is True
        assert result.response_time == 2.5

if __name__ == "__main__":
    pytest.main([__file__])
