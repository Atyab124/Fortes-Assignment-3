"""Database models and operations for the RAG application."""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from config import Config

Base = declarative_base()

class Document(Base):
    """Document model."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "content": self.content,
            "metadata": json.loads(self.metadata) if self.metadata else {},
            "created_at": self.created_at.isoformat()
        }

class Chunk(Base):
    """Text chunk model."""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    embedding_id = Column(String(50))  # Reference to vector DB
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat()
        }

class Query(Base):
    """Query log model."""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    response = Column(Text)
    citations = Column(Text)  # JSON string
    hallucination_detected = Column(Boolean, default=False)
    grounding_score = Column(Float)
    tokens_used = Column(Integer)
    estimated_cost = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "response": self.response,
            "citations": json.loads(self.citations) if self.citations else [],
            "hallucination_detected": self.hallucination_detected,
            "grounding_score": self.grounding_score,
            "tokens_used": self.tokens_used,
            "estimated_cost": self.estimated_cost,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat()
        }

class DatabaseManager:
    """Database manager for RAG application."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def add_document(self, filename: str, file_path: str, file_type: str, 
                    content: str, metadata: Dict[str, Any] = None) -> int:
        """Add a document to the database."""
        session = self.get_session()
        try:
            document = Document(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                content=content,
                metadata=json.dumps(metadata) if metadata else None
            )
            session.add(document)
            session.commit()
            return document.id
        finally:
            session.close()
    
    def add_chunk(self, document_id: int, chunk_index: int, content: str,
                 start_char: int, end_char: int, embedding_id: str = None) -> int:
        """Add a chunk to the database."""
        session = self.get_session()
        try:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                start_char=start_char,
                end_char=end_char,
                embedding_id=embedding_id
            )
            session.add(chunk)
            session.commit()
            return chunk.id
        finally:
            session.close()
    
    def add_query(self, query_text: str, response: str = None, 
                 citations: List[Dict] = None, hallucination_detected: bool = False,
                 grounding_score: float = None, tokens_used: int = None,
                 estimated_cost: float = None, processing_time: float = None) -> int:
        """Add a query to the database."""
        session = self.get_session()
        try:
            query = Query(
                query_text=query_text,
                response=response,
                citations=json.dumps(citations) if citations else None,
                hallucination_detected=hallucination_detected,
                grounding_score=grounding_score,
                tokens_used=tokens_used,
                estimated_cost=estimated_cost,
                processing_time=processing_time
            )
            session.add(query)
            session.commit()
            return query.id
        finally:
            session.close()
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        session = self.get_session()
        try:
            documents = session.query(Document).all()
            return [doc.to_dict() for doc in documents]
        finally:
            session.close()
    
    def get_chunks_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        session = self.get_session()
        try:
            chunks = session.query(Chunk).filter(Chunk.document_id == document_id).all()
            return [chunk.to_dict() for chunk in chunks]
        finally:
            session.close()
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries."""
        session = self.get_session()
        try:
            queries = session.query(Query).order_by(Query.created_at.desc()).limit(limit).all()
            return [query.to_dict() for query in queries]
        finally:
            session.close()
