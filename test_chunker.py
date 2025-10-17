"""Tests for document chunking functionality."""

import pytest
import tempfile
import os
from pathlib import Path

from document_processor import DocumentProcessor, ChunkProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_simple_text(self):
        """Test chunking of simple text."""
        text = "This is a test sentence. " * 20  # 20 sentences
        chunks = self.processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk['content']) <= 150 for chunk in chunks)  # Allow some margin
        
        # Check that chunks have required fields
        for chunk in chunks:
            assert 'index' in chunk
            assert 'content' in chunk
            assert 'start_line' in chunk
            assert 'end_line' in chunk
            assert 'metadata' in chunk
    
    def test_chunk_empty_text(self):
        """Test chunking of empty text."""
        chunks = self.processor.chunk_text("")
        assert len(chunks) == 0
    
    def test_chunk_whitespace_text(self):
        """Test chunking of whitespace-only text."""
        chunks = self.processor.chunk_text("   \n\n   ")
        assert len(chunks) == 0
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = self.processor.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]['content']
                next_chunk = chunks[i + 1]['content']
                
                # Should have some common words (simple overlap check)
                current_words = set(current_chunk.lower().split())
                next_words = set(next_chunk.lower().split())
                overlap = len(current_words.intersection(next_words))
                
                assert overlap > 0, f"No overlap between chunks {i} and {i+1}"
    
    def test_extract_txt_file(self):
        """Test extraction from .txt file."""
        test_content = "This is a test file content.\nWith multiple lines.\nAnd some text."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text(temp_path)
            assert "test file content" in extracted_text
            assert "multiple lines" in extracted_text
        finally:
            os.unlink(temp_path)
    
    def test_extract_markdown_file(self):
        """Test extraction from .md file."""
        test_content = """
# Test Markdown

This is a **bold** text with *italic* formatting.

## Section 2

- List item 1
- List item 2

```python
print("Hello, World!")
```
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            extracted_text = self.processor.extract_text(temp_path)
            # Should extract text without markdown formatting
            assert "Test Markdown" in extracted_text
            assert "bold" in extracted_text
            assert "italic" in extracted_text
            assert "List item" in extracted_text
        finally:
            os.unlink(temp_path)
    
    def test_process_file(self):
        """Test complete file processing."""
        test_content = "This is a test document. " * 10
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.processor.process_file(temp_path)
            
            assert 'filename' in result
            assert 'file_path' in result
            assert 'file_type' in result
            assert 'file_size' in result
            assert 'chunks' in result
            assert 'metadata' in result
            
            assert len(result['chunks']) > 0
            assert result['file_type'] == '.txt'
            assert result['filename'] == Path(temp_path).name
            
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.processor.extract_text(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.processor.extract_text("nonexistent_file.txt")

class TestChunkProcessor:
    """Test cases for ChunkProcessor."""
    
    def test_semantic_chunk(self):
        """Test semantic chunking."""
        text = "First topic sentence. Related sentence one. Related sentence two. Second topic sentence. Related sentence three."
        chunks = ChunkProcessor.semantic_chunk(text, chunk_size=50)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'content' in chunk
            assert 'length' in chunk
            assert len(chunk['content']) > 0
    
    def test_paragraph_chunk(self):
        """Test paragraph chunking."""
        text = "First paragraph with multiple sentences. This is the second sentence.\n\nSecond paragraph with different content. Another sentence here.\n\nThird paragraph with more content."
        chunks = ChunkProcessor.paragraph_chunk(text, chunk_size=50)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'content' in chunk
            assert 'length' in chunk
            assert len(chunk['content']) > 0
    
    def test_empty_text_chunking(self):
        """Test chunking of empty text."""
        semantic_chunks = ChunkProcessor.semantic_chunk("")
        paragraph_chunks = ChunkProcessor.paragraph_chunk("")
        
        assert len(semantic_chunks) == 0
        assert len(paragraph_chunks) == 0

if __name__ == "__main__":
    pytest.main([__file__])
