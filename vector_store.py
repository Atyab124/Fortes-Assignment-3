"""Vector storage and retrieval using FAISS."""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector storage for embeddings."""
    
    def __init__(self, dimension: int = 768, index_path: str = "faiss_index.bin"):
        """Initialize FAISS vector store."""
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.chunk_metadata = {}  # Maps FAISS IDs to chunk metadata
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                # Load metadata
                metadata_path = self.index_path.replace('.bin', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.chunk_metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Using IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_metadata = {}
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[int]:
        """Add vectors to the index with metadata."""
        if not vectors:
            return []
        
        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_array)
        
        # Add to index
        self.index.add(vectors_array)
        
        # Store metadata
        start_id = self.index.ntotal - len(vectors)
        for i, meta in enumerate(metadata):
            faiss_id = start_id + i
            self.chunk_metadata[faiss_id] = meta
        
        logger.info(f"Added {len(vectors)} vectors to index")
        return list(range(start_id, self.index.ntotal))
    
    def search(self, query_vector: List[float], k: int = 5, 
               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        similarities, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= similarity_threshold and idx in self.chunk_metadata:
                result = self.chunk_metadata[idx].copy()
                result['similarity'] = float(similarity)
                result['faiss_id'] = int(idx)
                results.append(result)
        
        logger.info(f"Found {len(results)} similar vectors above threshold {similarity_threshold}")
        return results
    
    def get_vector_by_id(self, faiss_id: int) -> Optional[List[float]]:
        """Get vector by FAISS ID."""
        if faiss_id >= self.index.ntotal:
            return None
        
        # Reconstruct vector from index
        vector = self.index.reconstruct(faiss_id)
        return vector.tolist()
    
    def get_metadata_by_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata by FAISS ID."""
        return self.chunk_metadata.get(faiss_id)
    
    def remove_vectors(self, faiss_ids: List[int]):
        """Remove vectors from the index."""
        # FAISS doesn't support direct removal, so we need to rebuild
        if not faiss_ids:
            return
        
        # Get all current vectors and metadata
        all_vectors = []
        new_metadata = {}
        new_id = 0
        
        for i in range(self.index.ntotal):
            if i not in faiss_ids:
                vector = self.index.reconstruct(i)
                all_vectors.append(vector)
                new_metadata[new_id] = self.chunk_metadata.get(i, {})
                new_id += 1
        
        # Rebuild index
        if all_vectors:
            self.index = faiss.IndexFlatIP(self.dimension)
            vectors_array = np.array(all_vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_array)
            self.index.add(vectors_array)
        else:
            self._create_new_index()
        
        self.chunk_metadata = new_metadata
        logger.info(f"Removed {len(faiss_ids)} vectors from index")
    
    def save_index(self):
        """Save the index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        metadata_path = self.index_path.replace('.bin', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP',
            'metadata_entries': len(self.chunk_metadata)
        }
    
    def clear_index(self):
        """Clear the entire index."""
        self._create_new_index()
        logger.info("Cleared FAISS index")

class VectorStoreManager:
    """Manages vector storage operations with database integration."""
    
    def __init__(self, vector_store: FAISSVectorStore, database):
        """Initialize vector store manager."""
        self.vector_store = vector_store
        self.db = database
    
    def add_document_vectors(self, document_id: int, chunks: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """Add document chunks and their embeddings."""
        # Add chunks to database
        chunk_ids = self.db.add_chunks(document_id, chunks)
        
        # Prepare metadata for FAISS
        metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk['metadata'].copy()
            chunk_metadata.update({
                'document_id': document_id,
                'chunk_id': chunk_ids[i],
                'chunk_index': chunk['index'],
                'content': chunk['content']
            })
            metadata_list.append(chunk_metadata)
        
        # Add embeddings to FAISS
        faiss_ids = self.vector_store.add_vectors(embeddings, metadata_list)
        
        # Update database with embedding references
        self.db.add_embeddings(chunk_ids, embeddings)
        
        logger.info(f"Added {len(chunks)} chunks and embeddings for document {document_id}")
        return chunk_ids, faiss_ids
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        results = self.vector_store.search(query_embedding, top_k, similarity_threshold)
        
        # Enhance results with database information
        enhanced_results = []
        for result in results:
            document = self.db.get_document(result['document_id'])
            if document:
                result['document'] = document
            enhanced_results.append(result)
        
        return enhanced_results
    
    def remove_document(self, document_id: int):
        """Remove all vectors for a document."""
        # Get all chunks for the document
        chunks = self.db.get_chunks_by_document(document_id)
        
        # Find FAISS IDs for these chunks
        faiss_ids_to_remove = []
        for chunk_id, metadata in self.vector_store.chunk_metadata.items():
            if metadata.get('document_id') == document_id:
                faiss_ids_to_remove.append(chunk_id)
        
        # Remove from FAISS
        if faiss_ids_to_remove:
            self.vector_store.remove_vectors(faiss_ids_to_remove)
        
        logger.info(f"Removed {len(faiss_ids_to_remove)} vectors for document {document_id}")
    
    def rebuild_index_from_database(self):
        """Rebuild FAISS index from database."""
        logger.info("Rebuilding FAISS index from database...")
        
        # Clear current index
        self.vector_store.clear_index()
        
        # Get all chunks with embeddings
        chunks = self.db.get_all_chunks_with_embeddings()
        
        if not chunks:
            logger.info("No chunks with embeddings found in database")
            return
        
        # Prepare data for FAISS
        embeddings = []
        metadata_list = []
        
        for chunk in chunks:
            if chunk['vector_data']:
                # Convert blob to vector
                vector = np.frombuffer(chunk['vector_data'], dtype=np.float32).tolist()
                embeddings.append(vector)
                
                # Prepare metadata
                metadata = chunk['metadata'].copy()
                metadata.update({
                    'document_id': chunk['document_id'],
                    'chunk_id': chunk['id'],
                    'chunk_index': chunk['chunk_index'],
                    'content': chunk['content']
                })
                metadata_list.append(metadata)
        
        # Add to FAISS
        if embeddings:
            self.vector_store.add_vectors(embeddings, metadata_list)
            logger.info(f"Rebuilt index with {len(embeddings)} vectors")
