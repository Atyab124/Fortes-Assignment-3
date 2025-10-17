"""Configuration settings for the RAG Q&A application."""

import os
from pathlib import Path
from typing import List, Optional

class Config:
    """Application configuration."""
    
    # Database settings
    DATABASE_URL = "sqlite:///./rag_app.db"
    VECTOR_DB_PATH = "./vector_db"
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Retrieval settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Generation settings
    MAX_TOKENS = 512
    TEMPERATURE = 0.7
    
    # Security settings
    ENABLE_GUARDRAILS = True
    PII_REDACTION = True
    
    # Evaluation settings
    EVAL_FILE = "eval.yaml"
    EVAL_OUTPUT = "eval_report.json"
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FILE = "rag_app.log"
    
    # Cost tracking
    ENABLE_COST_TRACKING = True
    PROMPT_CACHE_SIZE = 100
    
    # Supported file types
    SUPPORTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx"]
    
    # Sample corpus directory
    SAMPLE_CORPUS_DIR = "./sample_corpus"
    
    @classmethod
    def get_embedding_dim(cls) -> int:
        """Get embedding dimension for the selected model."""
        return 384  # all-MiniLM-L6-v2 dimension
    
    @classmethod
    def get_model_costs(cls) -> dict:
        """Get estimated costs per token for different models."""
        return {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "local": {"input": 0.0, "output": 0.0}
        }
