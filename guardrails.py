"""Safety guardrails for the RAG system."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for content filtering."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    UNSAFE = "unsafe"

class PromptInjectionDetector:
    """Detects prompt injection attempts."""
    
    def __init__(self):
        """Initialize prompt injection detector."""
        self.injection_patterns = [
            # Common injection patterns
            r'ignore\s+(?:previous|above|all)\s+(?:instructions?|prompts?)',
            r'forget\s+(?:everything|all|previous)',
            r'you\s+are\s+now\s+(?:a\s+)?(?:different|new)',
            r'act\s+as\s+if\s+you\s+are',
            r'pretend\s+to\s+be',
            r'roleplay\s+as',
            r'system\s*:\s*',
            r'assistant\s*:\s*',
            r'human\s*:\s*',
            r'user\s*:\s*',
            r'<\|.*?\|>',  # Special tokens
            r'\[INST\].*?\[/INST\]',  # Llama format
            r'<s>.*?</s>',  # BOS/EOS tokens
            r'###\s*(?:system|assistant|user)\s*:',
            r'---\s*(?:system|assistant|user)\s*:',
            r'```.*?```',  # Code blocks in prompts
            r'jailbreak',
            r'prompt\s+injection',
            r'override\s+instructions?',
            r'new\s+instructions?:\s*',
            r'ignore\s+the\s+above',
            r'disregard\s+previous',
            r'new\s+task:\s*',
            r'actually\s*,?\s*',
            r'nevermind\s*,?\s*',
            r'wait\s*,?\s*',
            r'hold\s+on\s*,?\s*',
            r'stop\s*,?\s*',
            r'cancel\s+that\s*,?\s*',
            r'disregard\s+everything',
            r'start\s+over',
            r'begin\s+again',
            r'reset\s+yourself',
            r'clear\s+your\s+memory',
            r'delete\s+everything',
            r'wipe\s+your\s+memory',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.injection_patterns]
    
    def detect(self, text: str) -> Tuple[SafetyLevel, List[str], Dict[str, Any]]:
        """Detect prompt injection attempts."""
        text_lower = text.lower()
        detected_patterns = []
        risk_score = 0.0
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text)
            if matches:
                detected_patterns.extend(matches)
                # Weight different patterns differently
                if i < 10:  # High-risk patterns
                    risk_score += 0.3
                elif i < 20:  # Medium-risk patterns
                    risk_score += 0.2
                else:  # Low-risk patterns
                    risk_score += 0.1
        
        # Additional heuristics
        if len(text) > 500 and '?' not in text and '!' not in text:
            risk_score += 0.1  # Suspiciously long without questions
        
        if text.count('"') > 10 or text.count("'") > 10:
            risk_score += 0.1  # Excessive quotes
        
        if any(word in text_lower for word in ['hack', 'exploit', 'bypass', 'trick']):
            risk_score += 0.2
        
        # Determine safety level
        if risk_score >= 0.7:
            safety_level = SafetyLevel.UNSAFE
        elif risk_score >= 0.3:
            safety_level = SafetyLevel.SUSPICIOUS
        else:
            safety_level = SafetyLevel.SAFE
        
        metadata = {
            'risk_score': risk_score,
            'pattern_count': len(detected_patterns),
            'text_length': len(text),
            'detected_patterns': detected_patterns[:5]  # Limit for logging
        }
        
        return safety_level, detected_patterns, metadata

class PIIRedactor:
    """Redacts personally identifiable information."""
    
    def __init__(self):
        """Initialize PII redactor."""
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            'mac_address': re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'),
            'date_of_birth': re.compile(r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'),
        }
        
        self.redaction_map = {
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'credit_card': '[CARD_REDACTED]',
            'ip_address': '[IP_REDACTED]',
            'url': '[URL_REDACTED]',
            'mac_address': '[MAC_REDACTED]',
            'date_of_birth': '[DOB_REDACTED]'
        }
    
    def redact(self, text: str, pii_types: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """Redact PII from text."""
        if pii_types is None:
            pii_types = list(self.pii_patterns.keys())
        
        redacted_text = text
        redacted_items = {}
        
        for pii_type in pii_types:
            if pii_type in self.pii_patterns:
                pattern = self.pii_patterns[pii_type]
                matches = pattern.findall(text)
                
                if matches:
                    redacted_items[pii_type] = len(matches)
                    redacted_text = pattern.sub(self.redaction_map[pii_type], redacted_text)
        
        metadata = {
            'original_length': len(text),
            'redacted_length': len(redacted_text),
            'redacted_items': redacted_items,
            'total_redacted': sum(redacted_items.values())
        }
        
        return redacted_text, metadata
    
    def detect_only(self, text: str) -> Dict[str, List[str]]:
        """Detect PII without redacting."""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii

class GroundingValidator:
    """Validates that answers are grounded in retrieved content."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize grounding validator."""
        self.similarity_threshold = similarity_threshold
    
    def validate(self, answer: str, retrieved_chunks: List[Dict[str, Any]], 
                question: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate if answer is grounded in retrieved content."""
        if not retrieved_chunks:
            return False, {'reason': 'no_retrieved_content', 'confidence': 0.0}
        
        # Check if we have high-quality retrieved content
        high_quality_chunks = [
            chunk for chunk in retrieved_chunks 
            if chunk.get('similarity', 0) >= self.similarity_threshold
        ]
        
        if not high_quality_chunks:
            return False, {
                'reason': 'low_similarity_scores',
                'max_similarity': max(chunk.get('similarity', 0) for chunk in retrieved_chunks),
                'confidence': 0.0
            }
        
        # Check answer length (very short answers might be suspicious)
        if len(answer.strip()) < 10:
            return False, {
                'reason': 'answer_too_short',
                'answer_length': len(answer.strip()),
                'confidence': 0.3
            }
        
        # Check if answer contains "I don't know" or similar phrases
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i can't", "i cannot",
            "no information", "not available", "unclear", "uncertain"
        ]
        
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
        
        if uncertainty_count > 2:
            return False, {
                'reason': 'high_uncertainty',
                'uncertainty_count': uncertainty_count,
                'confidence': 0.4
            }
        
        # Calculate grounding confidence
        max_similarity = max(chunk.get('similarity', 0) for chunk in high_quality_chunks)
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in high_quality_chunks) / len(high_quality_chunks)
        
        confidence = min(1.0, (max_similarity + avg_similarity) / 2)
        
        is_grounded = confidence >= 0.5 and len(high_quality_chunks) >= 1
        
        return is_grounded, {
            'reason': 'grounded' if is_grounded else 'low_confidence',
            'confidence': confidence,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'high_quality_chunks': len(high_quality_chunks),
            'total_chunks': len(retrieved_chunks)
        }

class ContentFilter:
    """Filters inappropriate or harmful content."""
    
    def __init__(self):
        """Initialize content filter."""
        self.harmful_patterns = [
            r'violence|violent|harm|hurt|kill|murder|assault',
            r'hate|racist|sexist|discriminat',
            r'terrorist|terrorism|bomb|explosive',
            r'drug|illegal|contraband',
            r'sexual|explicit|pornographic',
            r'suicide|self-harm|depression',
            r'fraud|scam|phishing|malware'
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.harmful_patterns]
    
    def filter(self, text: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Filter harmful content."""
        text_lower = text.lower()
        detected_patterns = []
        risk_score = 0.0
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                detected_patterns.extend(matches)
                risk_score += 0.2
        
        is_safe = risk_score < 0.3
        
        metadata = {
            'risk_score': risk_score,
            'detected_patterns': detected_patterns,
            'text_length': len(text)
        }
        
        return is_safe, detected_patterns, metadata

class SafetyGuardrails:
    """Main guardrails system."""
    
    def __init__(self, config=None):
        """Initialize safety guardrails."""
        self.config = config
        self.injection_detector = PromptInjectionDetector()
        self.pii_redactor = PIIRedactor()
        self.grounding_validator = GroundingValidator()
        self.content_filter = ContentFilter()
        
        # Cache for processed queries
        self.query_cache = {}
        self.cache_max_size = 1000
    
    def validate_query(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate user query before processing."""
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]
        
        # Clean query
        cleaned_query = query.strip()
        if not cleaned_query:
            result = (False, "Empty query", {'reason': 'empty_query'})
            self._cache_result(query_hash, result)
            return result
        
        # Check for prompt injection
        safety_level, patterns, injection_metadata = self.injection_detector.detect(cleaned_query)
        
        if safety_level == SafetyLevel.UNSAFE:
            result = (False, "Query contains potentially harmful prompt injection attempts", {
                'reason': 'prompt_injection',
                'safety_level': safety_level.value,
                'metadata': injection_metadata
            })
            self._cache_result(query_hash, result)
            return result
        
        # Check for harmful content
        is_safe, harmful_patterns, filter_metadata = self.content_filter.filter(cleaned_query)
        
        if not is_safe:
            result = (False, "Query contains inappropriate content", {
                'reason': 'harmful_content',
                'metadata': filter_metadata
            })
            self._cache_result(query_hash, result)
            return result
        
        # Redact PII from query
        redacted_query, pii_metadata = self.pii_redactor.redact(cleaned_query)
        
        # Cache result
        result = (True, redacted_query, {
            'original_query': cleaned_query,
            'redacted_query': redacted_query,
            'pii_metadata': pii_metadata,
            'injection_metadata': injection_metadata,
            'filter_metadata': filter_metadata
        })
        self._cache_result(query_hash, result)
        
        return result
    
    def validate_response(self, response: str, retrieved_chunks: List[Dict[str, Any]], 
                         original_query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate generated response."""
        # Check for prompt injection in response
        safety_level, patterns, injection_metadata = self.injection_detector.detect(response)
        
        if safety_level == SafetyLevel.UNSAFE:
            return False, "Response contains potentially harmful content", {
                'reason': 'harmful_response',
                'safety_level': safety_level.value,
                'metadata': injection_metadata
            }
        
        # Check for harmful content
        is_safe, harmful_patterns, filter_metadata = self.content_filter.filter(response)
        
        if not is_safe:
            return False, "Response contains inappropriate content", {
                'reason': 'inappropriate_response',
                'metadata': filter_metadata
            }
        
        # Validate grounding
        is_grounded, grounding_metadata = self.grounding_validator.validate(
            response, retrieved_chunks, original_query
        )
        
        if not is_grounded:
            return False, "Response is not properly grounded in retrieved content", {
                'reason': 'poor_grounding',
                'metadata': grounding_metadata
            }
        
        # Redact PII from response
        redacted_response, pii_metadata = self.pii_redactor.redact(response)
        
        return True, redacted_response, {
            'original_response': response,
            'redacted_response': redacted_response,
            'pii_metadata': pii_metadata,
            'grounding_metadata': grounding_metadata,
            'injection_metadata': injection_metadata,
            'filter_metadata': filter_metadata
        }
    
    def _cache_result(self, query_hash: str, result: Tuple[bool, str, Dict[str, Any]]):
        """Cache validation result."""
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query_hash] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        return {
            'cache_size': len(self.query_cache),
            'cache_max_size': self.cache_max_size,
            'pii_patterns': len(self.pii_redactor.pii_patterns),
            'injection_patterns': len(self.injection_detector.injection_patterns),
            'harmful_patterns': len(self.content_filter.harmful_patterns)
        }
