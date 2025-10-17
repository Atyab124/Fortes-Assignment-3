"""Document processing and chunking functionality."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document parsing and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self._extract_txt(file_path)
            elif file_extension == '.md':
                return self._extract_markdown(file_path)
            elif file_extension == '.pdf':
                return self._extract_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from .txt files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_markdown(self, file_path: Path) -> str:
        """Extract text from .md files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from .pdf files."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from .docx files."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(
                    current_chunk.strip(), 
                    chunk_index, 
                    metadata,
                    start_line=chunks[-1]['end_line'] + 1 if chunks else 1,
                    end_line=i
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += sentence
                current_length += sentence_length
        
        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                chunk_index,
                metadata,
                start_line=chunks[-1]['end_line'] + 1 if chunks else 1,
                end_line=len(sentences)
            ))
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', ' ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(chunk) <= self.chunk_overlap:
            return chunk
        
        # Find a good break point near the end
        overlap_start = len(chunk) - self.chunk_overlap
        overlap_text = chunk[overlap_start:]
        
        # Try to break at sentence boundary
        sentences = self._split_into_sentences(overlap_text)
        if len(sentences) > 1:
            # Take all but the last sentence
            overlap_text = ' '.join(sentences[:-1])
        
        return overlap_text + " "
    
    def _create_chunk(self, content: str, index: int, metadata: Optional[Dict], 
                     start_line: int, end_line: int) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            'chunk_size': len(content),
            'word_count': len(content.split()),
            'character_count': len(content)
        })
        
        return {
            'index': index,
            'content': content,
            'start_line': start_line,
            'end_line': end_line,
            'metadata': chunk_metadata
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file and return chunks with metadata."""
        file_path = Path(file_path)
        
        # Extract text
        text = self.extract_text(file_path)
        
        # Get file metadata
        file_stats = file_path.stat()
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_stats.st_size,
            'file_type': file_path.suffix,
            'word_count': len(text.split()),
            'character_count': len(text)
        }
        
        # Create chunks
        chunks = self.chunk_text(text, metadata)
        
        return {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_type': file_path.suffix,
            'file_size': file_stats.st_size,
            'text': text,
            'chunks': chunks,
            'metadata': metadata
        }

class ChunkProcessor:
    """Advanced chunking strategies."""
    
    @staticmethod
    def semantic_chunk(text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Semantic chunking based on content similarity."""
        # This is a simplified version - in practice, you'd use embeddings
        # to find semantically similar sentences and group them together
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'length': current_length
                })
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'length': current_length
            })
        
        return chunks
    
    @staticmethod
    def paragraph_chunk(text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Chunk by paragraphs."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_length = len(paragraph)
            
            if current_length + paragraph_length > chunk_size and current_chunk:
                chunks.append({
                    'content': '\n\n'.join(current_chunk),
                    'length': current_length
                })
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        if current_chunk:
            chunks.append({
                'content': '\n\n'.join(current_chunk),
                'length': current_length
            })
        
        return chunks
