"""Tests for document chunking functionality."""

import unittest
import tempfile
import os
from pathlib import Path

from document_processor import DocumentProcessor

class TestChunker(unittest.TestCase):
    """Test cases for document chunking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. " * 20  # Create long text
        chunks = self.processor.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk["content"] for chunk in chunks))
        self.assertTrue(all(chunk["start_char"] >= 0 for chunk in chunks))
        self.assertTrue(all(chunk["end_char"] > chunk["start_char"] for chunk in chunks))
    
    def test_chunk_text_overlap(self):
        """Test chunk overlap functionality."""
        text = "This is a test document. " * 20
        chunks = self.processor.chunk_text(text, chunk_size=50, chunk_overlap=10)
        
        # Check that chunks have proper overlap
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Check that there's some overlap in content
            prev_end = prev_chunk["content"][-10:]
            curr_start = curr_chunk["content"][:10]
            
            # There should be some overlap
            self.assertTrue(
                len(set(prev_end.split()) & set(curr_start.split())) > 0 or
                len(chunks) == 1  # Only one chunk
            )
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = self.processor.chunk_text("")
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        chunks = self.processor.chunk_text(text, chunk_size=100)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], text)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This   has    multiple    spaces.\n\nAnd newlines."
        cleaned = self.processor._clean_text(dirty_text)
        
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n", cleaned)
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.processor._split_into_sentences(text)
        
        self.assertEqual(len(sentences), 3)
        self.assertTrue(all(sentence.endswith('.') for sentence in sentences))
    
    def test_process_txt_file(self):
        """Test processing a text file."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test file.\n" * 10)
        
        # Process file
        result = self.processor.process_file(test_file)
        
        self.assertEqual(result["filename"], "test.txt")
        self.assertEqual(result["file_type"], ".txt")
        self.assertIn("This is a test file.", result["content"])
    
    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        # Create test markdown file
        test_file = os.path.join(self.temp_dir, "test.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("# Test Document\n\nThis is **bold** text.")
        
        # Process file
        result = self.processor.process_file(test_file)
        
        self.assertEqual(result["filename"], "test.md")
        self.assertEqual(result["file_type"], ".md")
        # Markdown should be converted to plain text
        self.assertIn("Test Document", result["content"])
        self.assertIn("bold", result["content"])
    
    def test_process_unsupported_file(self):
        """Test processing unsupported file type."""
        test_file = os.path.join(self.temp_dir, "test.xyz")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test content")
        
        with self.assertRaises(ValueError):
            self.processor.process_file(test_file)
    
    def test_process_nonexistent_file(self):
        """Test processing non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file("nonexistent.txt")
    
    def test_process_directory(self):
        """Test processing a directory."""
        # Create test files
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test{i}.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f"Test content {i}")
        
        # Process directory
        results = self.processor.process_directory(self.temp_dir)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["filename"].startswith("test") for result in results))
    
    def test_chunk_metadata(self):
        """Test chunk metadata."""
        text = "This is a test document. " * 20
        chunks = self.processor.chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk["chunk_index"], i)
            self.assertGreaterEqual(chunk["start_char"], 0)
            self.assertGreater(chunk["end_char"], chunk["start_char"])
            self.assertLessEqual(len(chunk["content"]), self.processor.chunk_size + 50)  # Allow some flexibility
    
    def test_chunk_boundaries(self):
        """Test chunk boundary handling."""
        text = "A. " * 100  # Many short sentences
        chunks = self.processor.chunk_text(text, chunk_size=50)
        
        # All chunks should be within size limit
        for chunk in chunks:
            self.assertLessEqual(len(chunk["content"]), 50 + 20)  # Allow some flexibility for overlap
    
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        text = "This is a test with émojis 🚀 and unicode characters."
        chunks = self.processor.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        self.assertIn("émojis", chunks[0]["content"])
        self.assertIn("🚀", chunks[0]["content"])

if __name__ == '__main__':
    unittest.main()
