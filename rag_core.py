"""Core RAG functionality with retrieval and generation."""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from config import Config
from vector_store import VectorStore
from database import DatabaseManager
from guardrails import Guardrails

@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    citations: List[Dict[str, Any]]
    hallucination_detected: bool
    grounding_score: float
    tokens_used: int
    estimated_cost: float
    processing_time: float
    query_id: Optional[int] = None

class RAGCore:
    """Core RAG system with retrieval and generation."""
    
    def __init__(self, use_local_model: bool = False, model_name: str = None):
        self.use_local_model = use_local_model
        self.model_name = model_name or ("gpt-3.5-turbo" if not use_local_model else "local")
        
        # Initialize components
        self.vector_store = VectorStore()
        self.db_manager = DatabaseManager()
        self.guardrails = Guardrails()
        
        # Initialize generation model
        self.generator = None
        self.tokenizer = None
        self._initialize_generator()
        
        # Prompt cache
        self.prompt_cache = {}
        self.cache_size = Config.PROMPT_CACHE_SIZE
    
    def _initialize_generator(self):
        """Initialize the text generation model."""
        if self.use_local_model:
            try:
                # Use a small local model for demo
                model_name = "microsoft/DialoGPT-small"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.generator = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            except Exception as e:
                print(f"Error loading local model: {e}")
                print("Falling back to OpenAI API")
                self.use_local_model = False
                self.model_name = "gpt-3.5-turbo"
        
        if not self.use_local_model:
            # Set OpenAI API key if available
            openai.api_key = "your-api-key-here"  # Should be set via environment variable
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest documents into the RAG system."""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        results = {
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": []
        }
        
        for file_path in file_paths:
            try:
                # Process document
                doc_data = processor.process_file(file_path)
                
                # Add to database
                doc_id = self.db_manager.add_document(
                    filename=doc_data["filename"],
                    file_path=doc_data["file_path"],
                    file_type=doc_data["file_type"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"]
                )
                
                # Chunk document
                chunks = processor.chunk_text(doc_data["content"])
                
                # Prepare chunks for vector store
                vector_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "chunk_id": f"{doc_id}_{i}",
                        "document_id": doc_id,
                        "chunk_index": i,
                        "content": chunk["content"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "filename": doc_data["filename"],
                        "file_type": doc_data["file_type"]
                    }
                    vector_chunks.append(chunk_data)
                    
                    # Add to database
                    self.db_manager.add_chunk(
                        document_id=doc_id,
                        chunk_index=i,
                        content=chunk["content"],
                        start_char=chunk["start_char"],
                        end_char=chunk["end_char"]
                    )
                
                # Add to vector store
                embedding_ids = self.vector_store.add_chunks(vector_chunks)
                
                # Update database with embedding IDs
                for i, embedding_id in enumerate(embedding_ids):
                    # This would require updating the database schema to store embedding IDs
                    pass
                
                results["documents_processed"] += 1
                results["chunks_created"] += len(chunks)
                
            except Exception as e:
                results["errors"].append(f"Error processing {file_path}: {str(e)}")
        
        return results
    
    def query(self, question: str, top_k: int = None, threshold: float = None) -> RAGResponse:
        """Query the RAG system."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{question}_{top_k}_{threshold}"
        if cache_key in self.prompt_cache:
            cached_response = self.prompt_cache[cache_key]
            cached_response.processing_time = time.time() - start_time
            return cached_response
        
        # Validate query with guardrails
        validation = self.guardrails.validate_query(question)
        if not validation["is_safe"]:
            return RAGResponse(
                answer=f"Query blocked: {validation['block_reason']}",
                citations=[],
                hallucination_detected=False,
                grounding_score=0.0,
                tokens_used=0,
                estimated_cost=0.0,
                processing_time=time.time() - start_time
            )
        
        # Retrieve relevant chunks
        top_k = top_k or Config.TOP_K
        threshold = threshold or Config.SIMILARITY_THRESHOLD
        
        retrieved_chunks = self.vector_store.search_with_threshold(
            question, threshold, top_k
        )
        
        if not retrieved_chunks:
            return RAGResponse(
                answer="No relevant information found in the knowledge base.",
                citations=[],
                hallucination_detected=False,
                grounding_score=0.0,
                tokens_used=0,
                estimated_cost=0.0,
                processing_time=time.time() - start_time
            )
        
        # Generate answer
        answer, citations, hallucination_detected = self._generate_answer(
            question, retrieved_chunks
        )
        
        # Calculate grounding score
        grounding_score = self._calculate_grounding_score(retrieved_chunks)
        
        # Calculate tokens and cost
        tokens_used = self._count_tokens(question + answer)
        estimated_cost = self._calculate_cost(tokens_used)
        
        # Create response
        response = RAGResponse(
            answer=answer,
            citations=citations,
            hallucination_detected=hallucination_detected,
            grounding_score=grounding_score,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            processing_time=time.time() - start_time
        )
        
        # Store in cache
        if len(self.prompt_cache) < self.cache_size:
            self.prompt_cache[cache_key] = response
        
        # Log query to database
        query_id = self.db_manager.add_query(
            query_text=question,
            response=answer,
            citations=citations,
            hallucination_detected=hallucination_detected,
            grounding_score=grounding_score,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            processing_time=response.processing_time
        )
        response.query_id = query_id
        
        return response
    
    def _generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], bool]:
        """Generate answer from retrieved chunks."""
        # Prepare context
        context = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        if self.use_local_model and self.generator:
            answer = self._generate_local(prompt)
        else:
            answer = self._generate_api(prompt)
        
        # Create citations
        citations = []
        for chunk in chunks:
            citation = {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "filename": chunk.get("filename", ""),
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "similarity_score": chunk["similarity_score"],
                "start_char": chunk.get("start_char", 0),
                "end_char": chunk.get("end_char", 0)
            }
            citations.append(citation)
        
        # Detect hallucination
        hallucination_detected = self._detect_hallucination(answer, chunks)
        
        return answer, citations, hallucination_detected
    
    def _generate_local(self, prompt: str) -> str:
        """Generate answer using local model."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            answer = response[len(prompt):].strip()
            return answer if answer else "I cannot generate a response with the current model."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_api(self, prompt: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _calculate_grounding_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate grounding score based on retrieved chunks."""
        if not chunks:
            return 0.0
        
        # Average similarity score
        avg_similarity = sum(chunk["similarity_score"] for chunk in chunks) / len(chunks)
        
        # Bonus for multiple chunks
        diversity_bonus = min(len(chunks) / 5.0, 0.2)  # Max 0.2 bonus
        
        return min(avg_similarity + diversity_bonus, 1.0)
    
    def _detect_hallucination(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        """Detect potential hallucination in the answer."""
        # Simple heuristic: check if answer contains information not in chunks
        chunk_text = " ".join([chunk["content"] for chunk in chunks]).lower()
        answer_lower = answer.lower()
        
        # Check for specific patterns that might indicate hallucination
        hallucination_indicators = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "i'm unable to",
            "i cannot find",
            "no information",
            "not available"
        ]
        
        # If answer contains these phrases, it's likely not hallucinated
        if any(indicator in answer_lower for indicator in hallucination_indicators):
            return False
        
        # Check if answer contains information not in chunks
        # This is a simplified check - in practice, you'd want more sophisticated methods
        answer_words = set(answer_lower.split())
        chunk_words = set(chunk_text.split())
        
        # If more than 30% of answer words are not in chunks, flag as potential hallucination
        unique_words = answer_words - chunk_words
        if len(unique_words) / len(answer_words) > 0.3:
            return True
        
        return False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.use_local_model and self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost for tokens."""
        if not Config.ENABLE_COST_TRACKING:
            return 0.0
        
        costs = Config.get_model_costs()
        model_costs = costs.get(self.model_name, costs["gpt-3.5-turbo"])
        
        # Assume 70% input, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        return (input_tokens * model_costs["input"] + output_tokens * model_costs["output"]) / 1000
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "cache_size": len(self.prompt_cache),
            "model_name": self.model_name,
            "use_local_model": self.use_local_model
        }
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self.prompt_cache.clear()
    
    def clear_all_data(self):
        """Clear all data from the system."""
        self.vector_store.clear()
        self.clear_cache()
        # Note: Database clearing would require additional implementation
