"""Database operations for storing document metadata and chunk mappings."""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VectorDatabase:
    """SQLite database for storing document metadata and chunk mappings."""
    
    def __init__(self, db_path: str = "vector_store.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_date TIMESTAMP,
                    metadata TEXT,
                    UNIQUE(file_path)
                )
            """)
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    embedding_id INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id INTEGER NOT NULL,
                    vector_data BLOB NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def add_document(self, filename: str, file_path: str, file_type: str, 
                    file_size: int, metadata: Optional[Dict] = None) -> int:
        """Add a document to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (filename, file_path, file_type, file_size, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, file_path, file_type, file_size, metadata_json))
            
            doc_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Added document: {filename} (ID: {doc_id})")
            return doc_id
    
    def add_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
        """Add chunks for a document."""
        chunk_ids = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks 
                    (document_id, chunk_index, content, start_line, end_line, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    chunk['index'],
                    chunk['content'],
                    chunk.get('start_line'),
                    chunk.get('end_line'),
                    json.dumps(chunk.get('metadata', {}))
                ))
                chunk_ids.append(cursor.lastrowid)
            
            conn.commit()
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            return chunk_ids
    
    def add_embeddings(self, chunk_ids: List[int], vectors: List[List[float]]):
        """Add embeddings for chunks."""
        import numpy as np
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for chunk_id, vector in zip(chunk_ids, vectors):
                # Convert vector to binary format
                vector_blob = np.array(vector, dtype=np.float32).tobytes()
                
                cursor.execute("""
                    INSERT INTO embeddings (chunk_id, vector_data)
                    VALUES (?, ?)
                """, (chunk_id, vector_blob))
                
                # Update chunk with embedding_id
                embedding_id = cursor.lastrowid
                cursor.execute("""
                    UPDATE chunks SET embedding_id = ? WHERE id = ?
                """, (embedding_id, chunk_id))
            
            conn.commit()
            logger.info(f"Added {len(vectors)} embeddings")
    
    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'filename': row[1],
                    'file_path': row[2],
                    'file_type': row[3],
                    'file_size': row[4],
                    'upload_date': row[5],
                    'processed_date': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {}
                }
            return None
    
    def get_chunks_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, chunk_index, content, start_line, end_line, 
                       embedding_id, metadata
                FROM chunks WHERE document_id = ? ORDER BY chunk_index
            """, (document_id,))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'chunk_index': row[1],
                    'content': row[2],
                    'start_line': row[3],
                    'end_line': row[4],
                    'embedding_id': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {}
                })
            return chunks
    
    def get_all_chunks_with_embeddings(self) -> List[Dict[str, Any]]:
        """Get all chunks with their embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.content, 
                       c.start_line, c.end_line, c.metadata, e.vector_data
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.vector_data IS NOT NULL
                ORDER BY c.document_id, c.chunk_index
            """)
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'document_id': row[1],
                    'chunk_index': row[2],
                    'content': row[3],
                    'start_line': row[4],
                    'end_line': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {},
                    'vector_data': row[7]
                })
            return chunks
    
    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get all chunks with embeddings
        chunks = self.get_all_chunks_with_embeddings()
        if not chunks:
            return []
        
        # Convert embeddings to numpy arrays
        query_vector = np.array(query_embedding).reshape(1, -1)
        chunk_vectors = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk['vector_data']:
                vector = np.frombuffer(chunk['vector_data'], dtype=np.float32)
                chunk_vectors.append(vector)
                valid_chunks.append(chunk)
        
        if not chunk_vectors:
            return []
        
        chunk_vectors = np.array(chunk_vectors)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = valid_chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Count embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = cursor.fetchone()[0]
            
            # Get file types
            cursor.execute("""
                SELECT file_type, COUNT(*) FROM documents 
                GROUP BY file_type
            """)
            file_types = dict(cursor.fetchall())
            
            return {
                'documents': doc_count,
                'chunks': chunk_count,
                'embeddings': embedding_count,
                'file_types': file_types or {}
            }
    
    def clear_database(self):
        """Clear all data from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embeddings")
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            conn.commit()
            logger.info("Database cleared")
