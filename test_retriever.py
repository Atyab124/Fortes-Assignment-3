"""Tests for retrieval functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from vector_store import FAISSVectorStore, VectorStoreManager
from database import VectorDatabase

class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        self.temp_file.close()
        
        self.vector_store = FAISSVectorStore(
            dimension=4,  # Small dimension for testing
            index_path=self.temp_file.name
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        if os.path.exists(self.temp_file.name.replace('.bin', '_metadata.pkl')):
            os.unlink(self.temp_file.name.replace('.bin', '_metadata.pkl'))
    
    def test_add_vectors(self):
        """Test adding vectors to the store."""
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
        
        metadata = [
            {'id': 1, 'content': 'First document'},
            {'id': 2, 'content': 'Second document'},
            {'id': 3, 'content': 'Third document'}
        ]
        
        faiss_ids = self.vector_store.add_vectors(vectors, metadata)
        
        assert len(faiss_ids) == 3
        assert self.vector_store.index.ntotal == 3
    
    def test_search_vectors(self):
        """Test searching for similar vectors."""
        # Add test vectors
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
        
        metadata = [
            {'id': 1, 'content': 'First document'},
            {'id': 2, 'content': 'Second document'},
            {'id': 3, 'content': 'Third document'}
        ]
        
        self.vector_store.add_vectors(vectors, metadata)
        
        # Search for similar vector
        query_vector = [0.9, 0.1, 0.0, 0.0]  # Similar to first vector
        results = self.vector_store.search(query_vector, k=2)
        
        assert len(results) > 0
        assert results[0]['id'] == 1  # Should find the first document
        assert results[0]['similarity'] > 0.5
    
    def test_search_empty_index(self):
        """Test searching in empty index."""
        query_vector = [1.0, 0.0, 0.0, 0.0]
        results = self.vector_store.search(query_vector, k=5)
        
        assert len(results) == 0
    
    def test_similarity_threshold(self):
        """Test similarity threshold filtering."""
        # Add test vectors
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]
        
        metadata = [
            {'id': 1, 'content': 'First document'},
            {'id': 2, 'content': 'Second document'}
        ]
        
        self.vector_store.add_vectors(vectors, metadata)
        
        # Search with high threshold
        query_vector = [0.1, 0.0, 0.0, 0.0]  # Low similarity to first vector
        results = self.vector_store.search(query_vector, k=5, similarity_threshold=0.9)
        
        assert len(results) == 0  # Should return no results due to high threshold
    
    def test_get_vector_by_id(self):
        """Test retrieving vector by ID."""
        vectors = [[1.0, 0.0, 0.0, 0.0]]
        metadata = [{'id': 1, 'content': 'Test document'}]
        
        faiss_ids = self.vector_store.add_vectors(vectors, metadata)
        
        retrieved_vector = self.vector_store.get_vector_by_id(faiss_ids[0])
        assert retrieved_vector is not None
        assert len(retrieved_vector) == 4
    
    def test_get_metadata_by_id(self):
        """Test retrieving metadata by ID."""
        vectors = [[1.0, 0.0, 0.0, 0.0]]
        metadata = [{'id': 1, 'content': 'Test document'}]
        
        faiss_ids = self.vector_store.add_vectors(vectors, metadata)
        
        retrieved_metadata = self.vector_store.get_metadata_by_id(faiss_ids[0])
        assert retrieved_metadata is not None
        assert retrieved_metadata['id'] == 1
        assert retrieved_metadata['content'] == 'Test document'
    
    def test_remove_vectors(self):
        """Test removing vectors."""
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]
        
        metadata = [
            {'id': 1, 'content': 'First document'},
            {'id': 2, 'content': 'Second document'}
        ]
        
        faiss_ids = self.vector_store.add_vectors(vectors, metadata)
        
        # Remove first vector
        self.vector_store.remove_vectors([faiss_ids[0]])
        
        assert self.vector_store.index.ntotal == 1
        
        # Verify remaining vector is the second one
        remaining_metadata = self.vector_store.get_metadata_by_id(0)
        assert remaining_metadata['id'] == 2
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        # Add some vectors
        vectors = [[1.0, 0.0, 0.0, 0.0]]
        metadata = [{'id': 1, 'content': 'Test document'}]
        
        self.vector_store.add_vectors(vectors, metadata)
        self.vector_store.save_index()
        
        # Create new vector store and load index
        new_vector_store = FAISSVectorStore(
            dimension=4,
            index_path=self.temp_file.name
        )
        
        assert new_vector_store.index.ntotal == 1
        assert len(new_vector_store.chunk_metadata) == 1
    
    def test_get_stats(self):
        """Test getting index statistics."""
        stats = self.vector_store.get_stats()
        
        assert 'total_vectors' in stats
        assert 'dimension' in stats
        assert 'index_type' in stats
        assert 'metadata_entries' in stats
        
        assert stats['dimension'] == 4
        assert stats['total_vectors'] == 0  # Initially empty
    
    def test_clear_index(self):
        """Test clearing the index."""
        # Add some vectors
        vectors = [[1.0, 0.0, 0.0, 0.0]]
        metadata = [{'id': 1, 'content': 'Test document'}]
        
        self.vector_store.add_vectors(vectors, metadata)
        assert self.vector_store.index.ntotal == 1
        
        # Clear index
        self.vector_store.clear_index()
        assert self.vector_store.index.ntotal == 0
        assert len(self.vector_store.chunk_metadata) == 0

class TestVectorDatabase:
    """Test cases for VectorDatabase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.db = VectorDatabase(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_add_document(self):
        """Test adding a document."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024,
            metadata={"author": "test"}
        )
        
        assert doc_id is not None
        assert doc_id > 0
    
    def test_get_document(self):
        """Test retrieving a document."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        document = self.db.get_document(doc_id)
        assert document is not None
        assert document['filename'] == "test.txt"
        assert document['file_type'] == ".txt"
    
    def test_add_chunks(self):
        """Test adding chunks."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [
            {
                'index': 0,
                'content': 'First chunk',
                'start_line': 1,
                'end_line': 5,
                'metadata': {'word_count': 2}
            },
            {
                'index': 1,
                'content': 'Second chunk',
                'start_line': 6,
                'end_line': 10,
                'metadata': {'word_count': 2}
            }
        ]
        
        chunk_ids = self.db.add_chunks(doc_id, chunks)
        
        assert len(chunk_ids) == 2
        assert all(cid > 0 for cid in chunk_ids)
    
    def test_get_chunks_by_document(self):
        """Test retrieving chunks for a document."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [
            {
                'index': 0,
                'content': 'First chunk',
                'start_line': 1,
                'end_line': 5
            }
        ]
        
        self.db.add_chunks(doc_id, chunks)
        retrieved_chunks = self.db.get_chunks_by_document(doc_id)
        
        assert len(retrieved_chunks) == 1
        assert retrieved_chunks[0]['content'] == 'First chunk'
    
    def test_add_embeddings(self):
        """Test adding embeddings."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [{'index': 0, 'content': 'Test chunk'}]
        chunk_ids = self.db.add_chunks(doc_id, chunks)
        
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        self.db.add_embeddings(chunk_ids, embeddings)
        
        # Verify embeddings were added
        chunks_with_embeddings = self.db.get_all_chunks_with_embeddings()
        assert len(chunks_with_embeddings) == 1
        assert chunks_with_embeddings[0]['vector_data'] is not None
    
    def test_search_similar_chunks(self):
        """Test searching for similar chunks."""
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [
            {'index': 0, 'content': 'First chunk'},
            {'index': 1, 'content': 'Second chunk'}
        ]
        chunk_ids = self.db.add_chunks(doc_id, chunks)
        
        # Add embeddings (similar vectors)
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]
        self.db.add_embeddings(chunk_ids, embeddings)
        
        # Search for similar chunks
        query_embedding = [0.9, 0.1, 0.0, 0.0]  # Similar to first embedding
        results = self.db.search_similar_chunks(query_embedding, top_k=2)
        
        assert len(results) > 0
        assert results[0]['similarity'] > 0.5
    
    def test_get_document_stats(self):
        """Test getting document statistics."""
        # Add some test data
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [{'index': 0, 'content': 'Test chunk'}]
        chunk_ids = self.db.add_chunks(doc_id, chunks)
        self.db.add_embeddings(chunk_ids, [[0.1, 0.2, 0.3, 0.4]])
        
        stats = self.db.get_document_stats()
        
        assert stats['documents'] == 1
        assert stats['chunks'] == 1
        assert stats['embeddings'] == 1
        assert '.txt' in stats['file_types']
    
    def test_clear_database(self):
        """Test clearing the database."""
        # Add some test data
        doc_id = self.db.add_document(
            filename="test.txt",
            file_path="/path/to/test.txt",
            file_type=".txt",
            file_size=1024
        )
        
        chunks = [{'index': 0, 'content': 'Test chunk'}]
        self.db.add_chunks(doc_id, chunks)
        
        # Clear database
        self.db.clear_database()
        
        stats = self.db.get_document_stats()
        assert stats['documents'] == 0
        assert stats['chunks'] == 0
        assert stats['embeddings'] == 0

if __name__ == "__main__":
    pytest.main([__file__])
