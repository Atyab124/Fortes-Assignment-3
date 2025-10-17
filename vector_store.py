"""Vector store implementation using FAISS."""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import json

from config import Config

class VectorStore:
    """FAISS-based vector store for embeddings."""
    
    def __init__(self, model_name: str = None, dimension: int = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.dimension = dimension or Config.get_embedding_dim()
        self.index = None
        self.embedder = None
        self.chunk_metadata = []
        self.index_path = f"{Config.VECTOR_DB_PATH}/faiss_index.bin"
        self.metadata_path = f"{Config.VECTOR_DB_PATH}/metadata.json"
        
        # Create vector DB directory if it doesn't exist
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
        else:
            self._create_index()
    
    def _create_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.embedder = SentenceTransformer(self.model_name)
        self.chunk_metadata = []
    
    def _load_index(self):
        """Load existing FAISS index."""
        try:
            self.index = faiss.read_index(self.index_path)
            self.embedder = SentenceTransformer(self.model_name)
            
            with open(self.metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
            self._create_index()
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            
            with open(self.metadata_path, 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add chunks to the vector store."""
        if not chunks:
            return []
        
        # Extract text content
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, convert_to_tensor=False)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        embedding_ids = []
        for i, chunk in enumerate(chunks):
            embedding_id = f"chunk_{len(self.chunk_metadata)}"
            embedding_ids.append(embedding_id)
            
            metadata = {
                "embedding_id": embedding_id,
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "content": chunk["content"],
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "filename": chunk.get("filename", ""),
                "file_type": chunk.get("file_type", "")
            }
            self.chunk_metadata.append(metadata)
        
        # Save index
        self._save_index()
        
        return embedding_ids
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index is None or len(self.chunk_metadata) == 0:
            return []
        
        top_k = top_k or Config.TOP_K
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result["similarity_score"] = float(score)
                results.append(result)
        
        return results
    
    def search_with_threshold(self, query: str, threshold: float = None, 
                            top_k: int = None) -> List[Dict[str, Any]]:
        """Search with similarity threshold."""
        results = self.search(query, top_k)
        threshold = threshold or Config.SIMILARITY_THRESHOLD
        
        return [r for r in results if r["similarity_score"] >= threshold]
    
    def get_chunk_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by embedding ID."""
        for metadata in self.chunk_metadata:
            if metadata["embedding_id"] == embedding_id:
                return metadata
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_chunks": len(self.chunk_metadata),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "model_name": self.model_name
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        self._create_index()
        self._save_index()
    
    def rebuild_index(self):
        """Rebuild the entire index from metadata."""
        if not self.chunk_metadata:
            return
        
        # Extract all texts
        texts = [metadata["content"] for metadata in self.chunk_metadata]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, convert_to_tensor=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save
        self._save_index()
