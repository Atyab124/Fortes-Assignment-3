"""Security guardrails for prompt injection detection and PII redaction."""

import re
from typing import List, Dict, Any, Tuple, Optional
import tiktoken

class Guardrails:
    """Security guardrails for the RAG application."""
    
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+(now|a)\s+(different|new)",
            r"pretend\s+to\s+be",
            r"act\s+as\s+if",
            r"roleplay\s+as",
            r"system\s*:\s*",
            r"assistant\s*:\s*",
            r"human\s*:\s*",
            r"jailbreak",
            r"bypass",
            r"override",
            r"hack",
            r"exploit",
            r"vulnerability",
            r"backdoor",
            r"admin\s+access",
            r"root\s+access",
            r"sudo",
            r"privilege\s+escalation",
            r"prompt\s+injection",
            r"data\s+exfiltration",
            r"unauthorized\s+access",
            r"security\s+breach",
            r"malicious\s+code",
            r"payload",
            r"exploit\s+code",
            r"injection\s+attack",
            r"social\s+engineering",
            r"phishing",
            r"malware",
            r"virus",
            r"trojan",
            r"backdoor",
            r"keylogger",
            r"spyware",
            r"ransomware",
            r"botnet",
            r"ddos",
            r"brute\s+force",
            r"dictionary\s+attack",
            r"rainbow\s+table",
            r"sql\s+injection",
            r"xss",
            r"csrf",
            r"buffer\s+overflow",
            r"format\s+string",
            r"integer\s+overflow",
            r"use\s+after\s+free",
            r"double\s+free",
            r"heap\s+spray",
            r"rop\s+chain",
            r"jop\s+chain",
            r"ret2libc",
            r"ret2syscall",
            r"aslr\s+bypass",
            r"dep\s+bypass",
            r"stack\s+canary\s+bypass",
            r"pie\s+bypass",
            r"relro\s+bypass",
            r"format\s+string\s+bug",
            r"heap\s+corruption",
            r"use\s+after\s+free",
            r"double\s+free",
            r"integer\s+overflow",
            r"buffer\s+overflow",
            r"stack\s+overflow",
            r"heap\s+overflow",
            r"format\s+string\s+vulnerability",
            r"integer\s+underflow",
            r"type\s+confusion",
            r"race\s+condition",
            r"time\s+of\s+check\s+time\s+of\s+use",
            r"toctou",
            r"symlink\s+attack",
            r"directory\s+traversal",
            r"path\s+traversal",
            r"file\s+inclusion",
            r"remote\s+file\s+inclusion",
            r"local\s+file\s+inclusion",
            r"lfi",
            r"rfi",
            r"command\s+injection",
            r"code\s+injection",
            r"ldap\s+injection",
            r"xpath\s+injection",
            r"no\s+sql\s+injection",
            r"nosql\s+injection",
            r"mongo\s+injection",
            r"couch\s+injection",
            r"cassandra\s+injection",
            r"redis\s+injection",
            r"memcached\s+injection",
            r"elasticsearch\s+injection",
            r"solr\s+injection",
            r"lucene\s+injection",
            r"hibernate\s+injection",
            r"jpa\s+injection",
            r"jdbc\s+injection",
            r"odbc\s+injection",
            r"ole\s+injection",
            r"com\s+injection",
            r"dcom\s+injection",
            r"corba\s+injection",
            r"soap\s+injection",
            r"rest\s+injection",
            r"graphql\s+injection",
            r"json\s+injection",
            r"xml\s+injection",
            r"yaml\s+injection",
            r"toml\s+injection",
            r"ini\s+injection",
            r"conf\s+injection",
            r"cfg\s+injection",
            r"properties\s+injection",
            r"env\s+injection",
            r"environment\s+injection",
            r"config\s+injection",
            r"setting\s+injection",
            r"parameter\s+injection",
            r"argument\s+injection",
            r"option\s+injection",
            r"flag\s+injection",
            r"switch\s+injection",
            r"variable\s+injection",
            r"value\s+injection",
            r"data\s+injection",
            r"input\s+injection",
            r"user\s+input\s+injection",
            r"form\s+injection",
            r"field\s+injection",
            r"attribute\s+injection",
            r"property\s+injection",
            r"element\s+injection",
            r"tag\s+injection",
            r"node\s+injection",
            r"object\s+injection",
            r"class\s+injection",
            r"method\s+injection",
            r"function\s+injection",
            r"procedure\s+injection",
            r"routine\s+injection",
            r"script\s+injection",
            r"code\s+injection",
            r"executable\s+injection",
            r"binary\s+injection",
            r"assembly\s+injection",
            r"machine\s+code\s+injection",
            r"bytecode\s+injection",
            r"opcode\s+injection",
            r"instruction\s+injection",
            r"command\s+injection",
            r"shell\s+injection",
            r"bash\s+injection",
            r"powershell\s+injection",
            r"cmd\s+injection",
            r"batch\s+injection",
            r"vbs\s+injection",
            r"vbscript\s+injection",
            r"jscript\s+injection",
            r"javascript\s+injection",
            r"python\s+injection",
            r"perl\s+injection",
            r"ruby\s+injection",
            r"php\s+injection",
            r"asp\s+injection",
            r"jsp\s+injection",
            r"servlet\s+injection",
            r"ejb\s+injection",
            r"bean\s+injection",
            r"component\s+injection",
            r"service\s+injection",
            r"api\s+injection",
            r"endpoint\s+injection",
            r"url\s+injection",
            r"uri\s+injection",
            r"path\s+injection",
            r"route\s+injection",
            r"handler\s+injection",
            r"controller\s+injection",
            r"action\s+injection",
            r"method\s+injection",
            r"function\s+injection",
            r"procedure\s+injection",
            r"routine\s+injection",
            r"script\s+injection",
            r"code\s+injection",
            r"executable\s+injection",
            r"binary\s+injection",
            r"assembly\s+injection",
            r"machine\s+code\s+injection",
            r"bytecode\s+injection",
            r"opcode\s+injection",
            r"instruction\s+injection",
            r"command\s+injection",
            r"shell\s+injection",
            r"bash\s+injection",
            r"powershell\s+injection",
            r"cmd\s+injection",
            r"batch\s+injection",
            r"vbs\s+injection",
            r"vbscript\s+injection",
            r"jscript\s+injection",
            r"javascript\s+injection",
            r"python\s+injection",
            r"perl\s+injection",
            r"ruby\s+injection",
            r"php\s+injection",
            r"asp\s+injection",
            r"jsp\s+injection",
            r"servlet\s+injection",
            r"ejb\s+injection",
            r"bean\s+injection",
            r"component\s+injection",
            r"service\s+injection",
            r"api\s+injection",
            r"endpoint\s+injection",
            r"url\s+injection",
            r"uri\s+injection",
            r"path\s+injection",
            r"route\s+injection",
            r"handler\s+injection",
            r"controller\s+injection",
            r"action\s+injection"
        ]
        
        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
            "date_of_birth": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
        
        # Compile patterns for efficiency
        self.injection_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
        self.pii_regex = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in self.pii_patterns.items()}
    
    def detect_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Detect potential prompt injection attempts."""
        detected_patterns = []
        
        for regex in self.injection_regex:
            if regex.search(text):
                detected_patterns.append(regex.pattern)
        
        return len(detected_patterns) > 0, detected_patterns
    
    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact PII from text."""
        redacted_text = text
        pii_counts = {}
        
        for pii_type, regex in self.pii_regex.items():
            matches = regex.findall(text)
            if matches:
                pii_counts[pii_type] = len(matches)
                if pii_type == "email":
                    redacted_text = regex.sub("[EMAIL_REDACTED]", redacted_text)
                elif pii_type == "phone":
                    redacted_text = regex.sub("[PHONE_REDACTED]", redacted_text)
                elif pii_type == "ssn":
                    redacted_text = regex.sub("[SSN_REDACTED]", redacted_text)
                elif pii_type == "credit_card":
                    redacted_text = regex.sub("[CARD_REDACTED]", redacted_text)
                elif pii_type == "ip_address":
                    redacted_text = regex.sub("[IP_REDACTED]", redacted_text)
                elif pii_type == "url":
                    redacted_text = regex.sub("[URL_REDACTED]", redacted_text)
                elif pii_type == "date_of_birth":
                    redacted_text = regex.sub("[DOB_REDACTED]", redacted_text)
        
        return redacted_text, pii_counts
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate a query for security issues."""
        # Check for prompt injection
        injection_detected, injection_patterns = self.detect_prompt_injection(query)
        
        # Check for PII
        pii_redacted, pii_counts = self.redact_pii(query)
        
        # Check query length
        is_too_long = len(query) > 1000
        
        # Check for empty query
        is_empty = not query.strip()
        
        # Determine if query should be blocked
        should_block = (
            injection_detected or 
            is_too_long or 
            is_empty or
            any(count > 0 for count in pii_counts.values())
        )
        
        return {
            "is_safe": not should_block,
            "injection_detected": injection_detected,
            "injection_patterns": injection_patterns,
            "pii_detected": any(count > 0 for count in pii_counts.values()),
            "pii_counts": pii_counts,
            "pii_redacted": pii_redacted,
            "is_too_long": is_too_long,
            "is_empty": is_empty,
            "should_block": should_block,
            "block_reason": self._get_block_reason(injection_detected, pii_counts, is_too_long, is_empty)
        }
    
    def _get_block_reason(self, injection_detected: bool, pii_counts: Dict[str, int], 
                         is_too_long: bool, is_empty: bool) -> str:
        """Get reason for blocking query."""
        if is_empty:
            return "Query is empty"
        elif is_too_long:
            return "Query is too long (max 1000 characters)"
        elif injection_detected:
            return "Potential prompt injection detected"
        elif any(count > 0 for count in pii_counts.values()):
            return "PII detected in query"
        else:
            return "Query is safe"
    
    def sanitize_response(self, response: str) -> Tuple[str, Dict[str, int]]:
        """Sanitize response for PII."""
        return self.redact_pii(response)
    
    def get_security_report(self, query: str, response: str = None) -> Dict[str, Any]:
        """Get comprehensive security report."""
        query_validation = self.validate_query(query)
        
        report = {
            "query_validation": query_validation,
            "response_sanitization": None
        }
        
        if response:
            sanitized_response, pii_counts = self.sanitize_response(response)
            report["response_sanitization"] = {
                "pii_detected": any(count > 0 for count in pii_counts.values()),
                "pii_counts": pii_counts,
                "sanitized_response": sanitized_response
            }
        
        return report
