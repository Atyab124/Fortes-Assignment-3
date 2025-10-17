"""Attribution and hallucination detection for RAG responses."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import difflib

logger = logging.getLogger(__name__)

class AttributionLevel(Enum):
    """Levels of attribution confidence."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass
class SentenceAttribution:
    """Attribution information for a sentence."""
    sentence: str
    sentence_index: int
    source_chunks: List[Dict[str, Any]]
    confidence: float
    attribution_level: AttributionLevel
    supporting_evidence: List[str]
    is_hallucinated: bool

class SentenceSplitter:
    """Splits text into sentences for attribution analysis."""
    
    def __init__(self):
        """Initialize sentence splitter."""
        # More sophisticated sentence splitting patterns
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$',
            re.MULTILINE
        )
        
        # Patterns that might indicate sentence boundaries
        self.boundary_patterns = [
            r'\.\s+[A-Z]',  # Period followed by capital letter
            r'!\s+[A-Z]',   # Exclamation followed by capital letter
            r'\?\s+[A-Z]',  # Question mark followed by capital letter
            r'\.\s*$',      # Period at end of string
            r'!\s*$',       # Exclamation at end of string
            r'\?\s*$'       # Question mark at end of string
        ]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

class SemanticMatcher:
    """Matches sentences to source chunks using semantic similarity."""
    
    def __init__(self):
        """Initialize semantic matcher."""
        self.similarity_threshold = 0.3
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Use difflib for basic similarity
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity
    
    def find_best_matches(self, sentence: str, source_chunks: List[Dict[str, Any]], 
                         top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Find best matching source chunks for a sentence."""
        matches = []
        
        for chunk in source_chunks:
            chunk_text = chunk.get('content', '')
            similarity = self.calculate_similarity(sentence, chunk_text)
            
            if similarity >= self.similarity_threshold:
                matches.append((chunk, similarity))
        
        # Sort by similarity and return top-k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def extract_supporting_evidence(self, sentence: str, chunk: Dict[str, Any]) -> List[str]:
        """Extract specific evidence from chunk that supports the sentence."""
        chunk_text = chunk.get('content', '')
        evidence = []
        
        # Split chunk into smaller segments for evidence extraction
        segments = chunk_text.split('. ')
        
        for segment in segments:
            segment = segment.strip()
            if len(segment) < 10:
                continue
                
            similarity = self.calculate_similarity(sentence, segment)
            if similarity >= 0.2:  # Lower threshold for evidence
                evidence.append(segment)
        
        return evidence[:3]  # Limit to top 3 evidence pieces

class HallucinationDetector:
    """Detects potential hallucinations in generated responses."""
    
    def __init__(self):
        """Initialize hallucination detector."""
        self.fact_indicators = [
            r'\b(?:according to|based on|as stated|as mentioned|as described)\b',
            r'\b(?:the document|the source|the text|it says)\b',
            r'\b(?:specifically|explicitly|clearly|directly)\b'
        ]
        
        self.uncertainty_indicators = [
            r'\b(?:might|may|could|possibly|perhaps|maybe)\b',
            r'\b(?:likely|probably|seems|appears)\b',
            r'\b(?:i think|i believe|i assume|i guess)\b'
        ]
        
        self.fact_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.fact_indicators]
        self.uncertainty_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.uncertainty_indicators]
    
    def detect_hallucination(self, sentence: str, source_chunks: List[Dict[str, Any]], 
                           semantic_matcher: SemanticMatcher) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect if a sentence is likely a hallucination."""
        # Check if sentence has any supporting evidence
        best_matches = semantic_matcher.find_best_matches(sentence, source_chunks, top_k=3)
        
        if not best_matches:
            # No supporting evidence found
            return True, 0.9, {
                'reason': 'no_supporting_evidence',
                'max_similarity': 0.0,
                'supporting_chunks': 0
            }
        
        max_similarity = best_matches[0][1]
        
        # Calculate hallucination probability
        hallucination_prob = 1.0 - max_similarity
        
        # Adjust based on linguistic patterns
        fact_mentions = sum(1 for pattern in self.fact_patterns if pattern.search(sentence))
        uncertainty_mentions = sum(1 for pattern in self.uncertainty_patterns if pattern.search(sentence))
        
        # More fact indicators = lower hallucination probability
        if fact_mentions > 0:
            hallucination_prob *= 0.7
        
        # More uncertainty indicators = higher hallucination probability
        if uncertainty_mentions > 0:
            hallucination_prob *= 1.2
        
        # Check for specific numbers, dates, or names without evidence
        specific_info = re.findall(r'\b\d{4}\b|\b\d+%\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence)
        if specific_info and max_similarity < 0.5:
            hallucination_prob += 0.2
        
        # Check sentence length (very long sentences might be more likely to contain hallucinations)
        if len(sentence.split()) > 50:
            hallucination_prob += 0.1
        
        hallucination_prob = min(1.0, hallucination_prob)
        is_hallucinated = hallucination_prob > 0.6
        
        return is_hallucinated, hallucination_prob, {
            'reason': 'low_similarity' if max_similarity < 0.4 else 'linguistic_patterns',
            'max_similarity': max_similarity,
            'supporting_chunks': len(best_matches),
            'fact_mentions': fact_mentions,
            'uncertainty_mentions': uncertainty_mentions,
            'specific_info_count': len(specific_info),
            'sentence_length': len(sentence.split())
        }

class AttributionAnalyzer:
    """Main attribution analyzer."""
    
    def __init__(self):
        """Initialize attribution analyzer."""
        self.sentence_splitter = SentenceSplitter()
        self.semantic_matcher = SemanticMatcher()
        self.hallucination_detector = HallucinationDetector()
    
    def analyze_response(self, response: str, source_chunks: List[Dict[str, Any]]) -> List[SentenceAttribution]:
        """Analyze response for attribution and hallucination."""
        sentences = self.sentence_splitter.split_sentences(response)
        attributions = []
        
        for i, sentence in enumerate(sentences):
            attribution = self._analyze_sentence(sentence, i, source_chunks)
            attributions.append(attribution)
        
        return attributions
    
    def _analyze_sentence(self, sentence: str, sentence_index: int, 
                         source_chunks: List[Dict[str, Any]]) -> SentenceAttribution:
        """Analyze a single sentence for attribution."""
        # Find best matching source chunks
        best_matches = self.semantic_matcher.find_best_matches(sentence, source_chunks)
        
        # Extract supporting evidence
        supporting_evidence = []
        source_chunk_refs = []
        
        for chunk, similarity in best_matches[:2]:  # Top 2 matches
            source_chunk_refs.append({
                'chunk_id': chunk.get('chunk_id'),
                'document_id': chunk.get('document_id'),
                'similarity': similarity,
                'filename': chunk.get('document', {}).get('filename', 'Unknown')
            })
            
            evidence = self.semantic_matcher.extract_supporting_evidence(sentence, chunk)
            supporting_evidence.extend(evidence)
        
        # Calculate overall confidence
        if best_matches:
            max_similarity = best_matches[0][1]
            avg_similarity = sum(match[1] for match in best_matches) / len(best_matches)
            confidence = (max_similarity + avg_similarity) / 2
        else:
            confidence = 0.0
        
        # Determine attribution level
        if confidence >= 0.7:
            attribution_level = AttributionLevel.HIGH
        elif confidence >= 0.4:
            attribution_level = AttributionLevel.MEDIUM
        elif confidence >= 0.2:
            attribution_level = AttributionLevel.LOW
        else:
            attribution_level = AttributionLevel.NONE
        
        # Detect hallucination
        is_hallucinated, hallucination_prob, hallucination_metadata = self.hallucination_detector.detect_hallucination(
            sentence, source_chunks, self.semantic_matcher
        )
        
        return SentenceAttribution(
            sentence=sentence,
            sentence_index=sentence_index,
            source_chunks=source_chunk_refs,
            confidence=confidence,
            attribution_level=attribution_level,
            supporting_evidence=supporting_evidence[:3],  # Limit evidence
            is_hallucinated=is_hallucinated
        )
    
    def generate_attribution_report(self, attributions: List[SentenceAttribution]) -> Dict[str, Any]:
        """Generate a comprehensive attribution report."""
        total_sentences = len(attributions)
        hallucinated_sentences = sum(1 for attr in attributions if attr.is_hallucinated)
        
        # Count by attribution level
        attribution_counts = {
            'high': sum(1 for attr in attributions if attr.attribution_level == AttributionLevel.HIGH),
            'medium': sum(1 for attr in attributions if attr.attribution_level == AttributionLevel.MEDIUM),
            'low': sum(1 for attr in attributions if attr.attribution_level == AttributionLevel.LOW),
            'none': sum(1 for attr in attributions if attr.attribution_level == AttributionLevel.NONE)
        }
        
        # Calculate average confidence
        avg_confidence = sum(attr.confidence for attr in attributions) / total_sentences if total_sentences > 0 else 0
        
        # Find problematic sentences
        problematic_sentences = [
            {
                'sentence': attr.sentence,
                'index': attr.sentence_index,
                'reason': 'hallucinated' if attr.is_hallucinated else 'low_attribution',
                'confidence': attr.confidence
            }
            for attr in attributions 
            if attr.is_hallucinated or attr.confidence < 0.3
        ]
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(attributions)
        
        return {
            'total_sentences': total_sentences,
            'hallucinated_sentences': hallucinated_sentences,
            'attribution_counts': attribution_counts,
            'average_confidence': avg_confidence,
            'quality_score': quality_score,
            'problematic_sentences': problematic_sentences,
            'attribution_summary': {
                'high_confidence': attribution_counts['high'],
                'medium_confidence': attribution_counts['medium'],
                'low_confidence': attribution_counts['low'],
                'no_attribution': attribution_counts['none']
            }
        }
    
    def _calculate_quality_score(self, attributions: List[SentenceAttribution]) -> float:
        """Calculate overall quality score for the response."""
        if not attributions:
            return 0.0
        
        # Weight different factors
        hallucination_penalty = sum(1 for attr in attributions if attr.is_hallucinated) / len(attributions)
        attribution_bonus = sum(attr.confidence for attr in attributions) / len(attributions)
        
        # Calculate score (0-1 scale)
        score = attribution_bonus - (hallucination_penalty * 0.5)
        return max(0.0, min(1.0, score))
    
    def highlight_problematic_sentences(self, response: str, attributions: List[SentenceAttribution]) -> str:
        """Highlight problematic sentences in the response."""
        highlighted_response = response
        
        for attr in attributions:
            if attr.is_hallucinated or attr.confidence < 0.3:
                # Add highlighting markers
                if attr.is_hallucinated:
                    marker = "🚨 [HALLUCINATED] "
                else:
                    marker = "⚠️ [LOW CONFIDENCE] "
                
                # Find and replace the sentence
                highlighted_response = highlighted_response.replace(
                    attr.sentence, 
                    marker + attr.sentence
                )
        
        return highlighted_response
