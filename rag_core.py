"""Core RAG functionality for document ingestion and querying."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from config import Config
from document_processor import DocumentProcessor
from vector_store import FAISSVectorStore, VectorStoreManager
from database import VectorDatabase

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system for document ingestion and querying."""
    
    def __init__(self, config: Config = None):
        """Initialize RAG system."""
        self.config = config or Config()
        self.db = VectorDatabase(self.config.VECTOR_DB_PATH)
        self.vector_store = FAISSVectorStore(
            dimension=self.config.EMBEDDING_DIMENSION,
            index_path=self.config.FAISS_INDEX_PATH
        )
        self.vector_manager = VectorStoreManager(self.vector_store, self.db)
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.ollama_client = None
        self._init_ollama()
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            from ollama import Client
            self.ollama_client = Client(host=self.config.OLLAMA_BASE_URL)
            
            # Test connection
            models = self.ollama_client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.config.EMBEDDING_MODEL not in available_models:
                logger.warning(f"Embedding model {self.config.EMBEDDING_MODEL} not found")
            if self.config.CHAT_MODEL not in available_models:
                logger.warning(f"Chat model {self.config.CHAT_MODEL} not found")
            
            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document."""
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Process document
            doc_data = self.doc_processor.process_file(str(file_path))
            
            # Add document to database
            doc_id = self.db.add_document(
                filename=doc_data['filename'],
                file_path=doc_data['file_path'],
                file_type=doc_data['file_type'],
                file_size=doc_data['file_size'],
                metadata=doc_data['metadata']
            )
            
            # Generate embeddings for chunks
            embeddings = self._generate_embeddings([chunk['content'] for chunk in doc_data['chunks']])
            
            # Add to vector store
            chunk_ids, faiss_ids = self.vector_manager.add_document_vectors(
                doc_id, doc_data['chunks'], embeddings
            )
            
            # Save vector store
            self.vector_store.save_index()
            
            processing_time = time.time() - start_time
            
            result = {
                'document_id': doc_id,
                'filename': doc_data['filename'],
                'chunks_created': len(doc_data['chunks']),
                'embeddings_generated': len(embeddings),
                'processing_time': processing_time,
                'success': True
            }
            
            logger.info(f"Successfully ingested {doc_data['filename']} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            return {
                'filename': doc_data['filename'] if 'doc_data' in locals() else file_path.name,
                'error': str(e),
                'success': False
            }
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """Ingest all supported documents in a directory."""
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        results = []
        total_files = 0
        successful_files = 0
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.config.SUPPORTED_EXTENSIONS:
                total_files += 1
                result = self.ingest_document(str(file_path))
                results.append(result)
                if result['success']:
                    successful_files += 1
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': total_files - successful_files,
            'results': results
        }
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using Ollama."""
        try:
            from ollama import embed
            
            embeddings = []
            batch_size = 10  # Process in batches to avoid memory issues
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = embed(
                    model=self.config.EMBEDDING_MODEL,
                    input=batch
                )
                
                # Extract embeddings from response
                if isinstance(response, dict) and 'embeddings' in response:
                    batch_embeddings = response['embeddings']
                else:
                    batch_embeddings = response
                
                embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def query(self, question: str, top_k: int = None, 
              similarity_threshold: float = None) -> Dict[str, Any]:
        """Query the RAG system."""
        start_time = time.time()
        
        top_k = top_k or self.config.TOP_K
        similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
        
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([question])[0]
            
            # Search for similar chunks
            similar_chunks = self.vector_manager.search_similar(
                query_embedding, top_k, similarity_threshold
            )
            
            if not similar_chunks:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'similarity_scores': [],
                    'query_time': time.time() - start_time,
                    'retrieval_success': False
                }
            
            # Generate answer using retrieved context
            answer = self._generate_answer(question, similar_chunks)
            
            # Prepare sources
            sources = []
            similarity_scores = []
            for chunk in similar_chunks:
                sources.append({
                    'document_id': chunk['document_id'],
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                    'filename': chunk.get('document', {}).get('filename', 'Unknown'),
                    'similarity': chunk['similarity']
                })
                similarity_scores.append(chunk['similarity'])
            
            query_time = time.time() - start_time
            
            result = {
                'answer': answer,
                'sources': sources,
                'similarity_scores': similarity_scores,
                'query_time': query_time,
                'retrieval_success': True,
                'num_sources': len(sources)
            }
            
            logger.info(f"Query processed in {query_time:.2f}s with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}",
                'sources': [],
                'similarity_scores': [],
                'query_time': time.time() - start_time,
                'retrieval_success': False,
                'error': str(e)
            }
    
    def _generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context."""
        try:
            from ollama import chat
            
            # Prepare context
            context = "\n\n".join([
                f"Source {i+1}:\n{chunk['content']}" 
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create system prompt
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer the question. If the context doesn't contain 
            enough information to answer the question, say so clearly. Be concise and accurate."""
            
            # Create user prompt
            user_prompt = f"""Based on the following context, please answer the question: {question}

Context:
{context}"""
            
            # Generate response
            response = chat(
                model=self.config.CHAT_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I encountered an error generating an answer: {str(e)}"
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents."""
        db_stats = self.db.get_document_stats()
        vector_stats = self.vector_store.get_stats()
        
        return {
            'database': db_stats,
            'vector_store': vector_stats,
            'total_documents': db_stats['documents'],
            'total_chunks': db_stats['chunks'],
            'total_embeddings': db_stats['embeddings']
        }
    
    def clear_all_data(self):
        """Clear all ingested data."""
        self.db.clear_database()
        self.vector_store.clear_index()
        logger.info("Cleared all data from RAG system")
    
    def rebuild_index(self):
        """Rebuild vector index from database."""
        self.vector_manager.rebuild_index_from_database()
        self.vector_store.save_index()
        logger.info("Rebuilt vector index from database")

class RAGIngestionPipeline:
    """Pipeline for batch document ingestion."""
    
    def __init__(self, rag_system: RAGSystem):
        """Initialize ingestion pipeline."""
        self.rag_system = rag_system
        self.processed_files = []
        self.failed_files = []
    
    def ingest_batch(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest multiple files."""
        results = []
        
        for file_path in file_paths:
            result = self.rag_system.ingest_document(file_path)
            results.append(result)
            
            if result['success']:
                self.processed_files.append(file_path)
            else:
                self.failed_files.append(file_path)
        
        return {
            'total_files': len(file_paths),
            'successful': len(self.processed_files),
            'failed': len(self.failed_files),
            'results': results
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """Get ingestion progress."""
        total = len(self.processed_files) + len(self.failed_files)
        return {
            'total_processed': total,
            'successful': len(self.processed_files),
            'failed': len(self.failed_files),
            'success_rate': len(self.processed_files) / total if total > 0 else 0
        }
