"""Document processing and chunking functionality."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

from config import Config

class DocumentProcessor:
    """Handles document ingestion and chunking."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return content with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = file_path.suffix.lower()
        
        if file_type == '.txt':
            content = self._process_txt(file_path)
        elif file_type == '.md':
            content = self._process_markdown(file_path)
        elif file_type == '.pdf':
            content = self._process_pdf(file_path)
        elif file_type == '.docx':
            content = self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_type": file_type,
            "content": content,
            "metadata": {
                "file_size": file_path.stat().st_size,
                "processed_at": str(Path(file_path).stat().st_mtime)
            }
        }
    
    def _process_txt(self, file_path: Path) -> str:
        """Process a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _process_markdown(self, file_path: Path) -> str:
        """Process a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process a PDF file."""
        content = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        return content
    
    def _process_docx(self, file_path: Path) -> str:
        """Process a Word document."""
        doc = docx.Document(file_path)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content
    
    def chunk_text(self, text: str, chunk_size: int = None, 
                  chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": current_chunk.strip(),
                    "start_char": current_start,
                    "end_char": current_start + len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_index += 1
            else:
                current_chunk += sentence
        
        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append({
                "chunk_index": chunk_index,
                "content": current_chunk.strip(),
                "start_char": current_start,
                "end_char": current_start + len(current_chunk)
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all supported files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        processed_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in Config.SUPPORTED_EXTENSIONS:
                try:
                    file_data = self.process_file(str(file_path))
                    processed_files.append(file_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return processed_files
