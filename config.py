"""Configuration settings for the RAG Q&A App."""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration."""
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    CHAT_MODEL = "qwen2.5:latest"
    
    # Vector store settings
    VECTOR_DB_PATH = "vector_store.db"
    FAISS_INDEX_PATH = "faiss_index.bin"
    EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension
    
    # Document processing
    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.pdf', '.docx'}
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Guardrails
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
    }
    
    # Evaluation
    EVAL_CONFIG_PATH = "eval.yaml"
    EVAL_REPORT_PATH = "eval_report.json"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # UI settings
    STREAMLIT_PORT = 8501
    STREAMLIT_HOST = "localhost"
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    # @classmethod
    # def get_sample_corpus_dir(cls) -> Path:
    #     """Get the sample corpus directory path."""
    #     corpus_dir = Path("sample_corpus")
    #     corpus_dir.mkdir(exist_ok=True)
    #     return corpus_dir

    @classmethod
    def get_sample_corpus_dir(cls) -> Path:
        """Get absolute path to sample corpus directory."""
        base_dir = Path(__file__).resolve().parent
        corpus_dir = base_dir / "sample_corpus"
        corpus_dir.mkdir(exist_ok=True)
        return corpus_dir
