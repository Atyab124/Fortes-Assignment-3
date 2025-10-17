"""Tests for retrieval functionality."""

import unittest
import tempfile
import os
import numpy as np

from vector_store import VectorStore
from rag_core import RAGCore

class TestRetriever(unittest.TestCase):
    """Test cases for retrieval functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore()
        
        # Create test chunks
        self.test_chunks = [
            {
                "chunk_id": "test_1",
                "document_id": 1,
                "chunk_index": 0,
                "content": "This is about machine learning and artificial intelligence.",
                "start_char": 0,
                "end_char": 50,
                "filename": "test1.md",
                "file_type": ".md"
            },
            {
                "chunk_id": "test_2",
                "document_id": 1,
                "chunk_index": 1,
                "content": "Deep learning is a subset of machine learning.",
                "start_char": 51,
                "end_char": 100,
                "filename": "test1.md",
                "file_type": ".md"
            },
            {
                "chunk_id": "test_3",
                "document_id": 2,
                "chunk_index": 0,
                "content": "Natural language processing uses neural networks.",
                "start_char": 0,
                "end_char": 50,
                "filename": "test2.md",
                "file_type": ".md"
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.vector_store.clear()
    
    def test_add_chunks(self):
        """Test adding chunks to vector store."""
        embedding_ids = self.vector_store.add_chunks(self.test_chunks)
        
        self.assertEqual(len(embedding_ids), len(self.test_chunks))
        self.assertTrue(all(embedding_id for embedding_id in embedding_ids))
    
    def test_search_basic(self):
        """Test basic search functionality."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Search
        results = self.vector_store.search("machine learning", top_k=2)
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)
        
        # Check result structure
        for result in results:
            self.assertIn("content", result)
            self.assertIn("similarity_score", result)
            self.assertIn("chunk_id", result)
            self.assertGreaterEqual(result["similarity_score"], 0.0)
            self.assertLessEqual(result["similarity_score"], 1.0)
    
    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Search with high threshold
        results = self.vector_store.search_with_threshold("machine learning", threshold=0.9, top_k=5)
        
        # Should return fewer results with high threshold
        self.assertLessEqual(len(results), len(self.test_chunks))
        
        # All results should meet threshold
        for result in results:
            self.assertGreaterEqual(result["similarity_score"], 0.9)
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        self.vector_store.add_chunks(self.test_chunks)
        
        results = self.vector_store.search("")
        self.assertEqual(len(results), 0)
    
    def test_search_no_chunks(self):
        """Test search when no chunks are added."""
        results = self.vector_store.search("test query")
        self.assertEqual(len(results), 0)
    
    def test_get_chunk_by_id(self):
        """Test getting chunk by embedding ID."""
        embedding_ids = self.vector_store.add_chunks(self.test_chunks)
        
        # Get chunk by ID
        chunk = self.vector_store.get_chunk_by_id(embedding_ids[0])
        
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["chunk_id"], self.test_chunks[0]["chunk_id"])
        self.assertEqual(chunk["content"], self.test_chunks[0]["content"])
    
    def test_get_chunk_by_nonexistent_id(self):
        """Test getting chunk by non-existent ID."""
        chunk = self.vector_store.get_chunk_by_id("nonexistent_id")
        self.assertIsNone(chunk)
    
    def test_get_stats(self):
        """Test getting vector store statistics."""
        initial_stats = self.vector_store.get_stats()
        self.assertEqual(initial_stats["total_chunks"], 0)
        
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        stats = self.vector_store.get_stats()
        self.assertEqual(stats["total_chunks"], len(self.test_chunks))
        self.assertGreater(stats["index_size"], 0)
    
    def test_clear(self):
        """Test clearing vector store."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Verify chunks are added
        stats = self.vector_store.get_stats()
        self.assertGreater(stats["total_chunks"], 0)
        
        # Clear
        self.vector_store.clear()
        
        # Verify chunks are removed
        stats = self.vector_store.get_stats()
        self.assertEqual(stats["total_chunks"], 0)
    
    def test_rebuild_index(self):
        """Test rebuilding index."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Rebuild index
        self.vector_store.rebuild_index()
        
        # Verify search still works
        results = self.vector_store.search("machine learning")
        self.assertGreater(len(results), 0)
    
    def test_search_similarity_ordering(self):
        """Test that search results are ordered by similarity."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Search
        results = self.vector_store.search("machine learning", top_k=3)
        
        # Results should be ordered by similarity (descending)
        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i-1]["similarity_score"],
                results[i]["similarity_score"]
            )
    
    def test_search_multiple_queries(self):
        """Test multiple searches."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Multiple searches
        queries = ["machine learning", "deep learning", "neural networks"]
        
        for query in queries:
            results = self.vector_store.search(query, top_k=2)
            self.assertLessEqual(len(results), 2)
            
            # Each result should have required fields
            for result in results:
                self.assertIn("content", result)
                self.assertIn("similarity_score", result)
                self.assertIn("chunk_id", result)
    
    def test_chunk_metadata_preservation(self):
        """Test that chunk metadata is preserved."""
        # Add chunks
        embedding_ids = self.vector_store.add_chunks(self.test_chunks)
        
        # Search and verify metadata
        results = self.vector_store.search("machine learning")
        
        for result in results:
            self.assertIn("chunk_id", result)
            self.assertIn("document_id", result)
            self.assertIn("filename", result)
            self.assertIn("file_type", result)
            self.assertIn("start_char", result)
            self.assertIn("end_char", result)
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent."""
        # Add chunks
        self.vector_store.add_chunks(self.test_chunks)
        
        # Search multiple times with same query
        query = "machine learning"
        results1 = self.vector_store.search(query)
        results2 = self.vector_store.search(query)
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1["chunk_id"], r2["chunk_id"])
            self.assertAlmostEqual(r1["similarity_score"], r2["similarity_score"], places=5)
    
    def test_large_chunk_handling(self):
        """Test handling of large chunks."""
        large_chunk = {
            "chunk_id": "large_1",
            "document_id": 1,
            "chunk_index": 0,
            "content": "This is a very long chunk. " * 100,  # Very long content
            "start_char": 0,
            "end_char": 2500,
            "filename": "large.md",
            "file_type": ".md"
        }
        
        embedding_ids = self.vector_store.add_chunks([large_chunk])
        self.assertEqual(len(embedding_ids), 1)
        
        # Search should still work
        results = self.vector_store.search("long chunk")
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
