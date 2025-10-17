"""Web UI for the RAG application using Streamlit."""

import streamlit as st
import time
import json
from typing import List, Dict, Any
from datetime import datetime

from rag_core import RAGCore
from database import DatabaseManager
from observability import ObservabilityManager
from evaluation import RAGEvaluator

class RAGWebUI:
    """Web UI for the RAG application."""
    
    def __init__(self):
        self.rag_core = None
        self.db_manager = None
        self.observability = None
        self.evaluator = None
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def initialize_rag(self, use_local_model: bool = True):
        """Initialize RAG system."""
        if not st.session_state.rag_initialized:
            with st.spinner("Initializing RAG system..."):
                try:
                    self.rag_core = RAGCore(use_local_model=use_local_model)
                    self.db_manager = DatabaseManager()
                    self.observability = ObservabilityManager()
                    self.evaluator = RAGEvaluator(self.rag_core)
                    
                    # Create database tables
                    self.db_manager.create_tables()
                    
                    st.session_state.rag_initialized = True
                    st.success("RAG system initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize RAG system: {str(e)}")
                    return False
        return True
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        with st.sidebar:
            st.title("RAG Q&A App")
            
            # Model selection
            st.subheader("Model Settings")
            use_local = st.checkbox("Use Local Model", value=True, help="Use local model instead of API")
            
            # Initialize RAG system
            if st.button("Initialize RAG System"):
                self.initialize_rag(use_local_model=use_local)
            
            # File upload
            st.subheader("Document Upload")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=['md', 'txt', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload documents to add to the knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Documents"):
                    self.process_uploaded_files(uploaded_files)
            
            # Evaluation
            st.subheader("Evaluation")
            if st.button("Run Evaluation"):
                self.run_evaluation()
            
            # Statistics
            st.subheader("Statistics")
            self.render_statistics()
            
            # Settings
            st.subheader("Settings")
            self.render_settings()
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files."""
        if not st.session_state.rag_initialized:
            st.error("Please initialize the RAG system first.")
            return
        
        with st.spinner("Processing documents..."):
            try:
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(temp_path)
                
                # Process documents
                results = self.rag_core.ingest_documents(temp_files)
                
                # Clean up temp files
                import os
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                st.success(f"Processed {results['documents_processed']} documents, created {results['chunks_created']} chunks")
                
                if results['errors']:
                    st.warning(f"Errors: {len(results['errors'])}")
                    for error in results['errors']:
                        st.error(error)
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    def run_evaluation(self):
        """Run evaluation."""
        if not st.session_state.rag_initialized:
            st.error("Please initialize the RAG system first.")
            return
        
        with st.spinner("Running evaluation..."):
            try:
                # Load evaluation data
                eval_data = self.evaluator.load_eval_data()
                
                if not eval_data:
                    st.warning("No evaluation data found. Creating sample data.")
                    return
                
                # Run evaluation
                results = self.evaluator.run_evaluation(eval_data)
                
                # Display results
                st.subheader("Evaluation Results")
                
                overall_metrics = results["overall_metrics"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Exact Match Rate", f"{overall_metrics['exact_match_rate']:.3f}")
                with col2:
                    st.metric("Average F1 Score", f"{overall_metrics['average_f1_score']:.3f}")
                with col3:
                    st.metric("Citation Accuracy", f"{overall_metrics['average_citation_accuracy']:.3f}")
                
                # Display detailed results
                if st.checkbox("Show Detailed Results"):
                    for result in results["detailed_results"]:
                        with st.expander(f"Question {result['id']}: {result['question'][:50]}..."):
                            st.write(f"**Expected:** {result['expected_answer']}")
                            st.write(f"**Actual:** {result['actual_answer']}")
                            st.write(f"**Exact Match:** {result['metrics']['exact_match']}")
                            st.write(f"**F1 Score:** {result['metrics']['f1_score']:.3f}")
                            st.write(f"**Similarity:** {result['metrics']['similarity_score']:.3f}")
                            st.write(f"**Citation Accuracy:** {result['metrics']['citation_accuracy']:.3f}")
                            st.write(f"**Hallucination Detected:** {result['metrics']['hallucination_detected']}")
                
                # Save results
                output_file = self.evaluator.save_results()
                st.success(f"Evaluation results saved to {output_file}")
                
            except Exception as e:
                st.error(f"Error running evaluation: {str(e)}")
    
    def render_statistics(self):
        """Render statistics."""
        if not st.session_state.rag_initialized:
            st.info("Initialize RAG system to see statistics")
            return
        
        try:
            # Get RAG stats
            rag_stats = self.rag_core.get_stats()
            st.write(f"**Vector Store:** {rag_stats['vector_store_stats']['total_chunks']} chunks")
            st.write(f"**Model:** {rag_stats['model_name']}")
            st.write(f"**Cache Size:** {rag_stats['cache_size']}")
            
            # Get cost summary
            if self.observability:
                cost_summary = self.observability.get_cost_summary()
                st.write(f"**Total Cost:** ${cost_summary['total_cost']:.4f}")
                st.write(f"**Total Queries:** {cost_summary['total_queries']}")
                
                if cost_summary['total_queries'] > 0:
                    st.write(f"**Avg Cost/Query:** ${cost_summary['average_cost_per_query']:.4f}")
            
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")
    
    def render_settings(self):
        """Render settings."""
        # Top K setting
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
        
        # Similarity threshold
        threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Max tokens
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=512)
        
        # Temperature
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        
        return {
            "top_k": top_k,
            "threshold": threshold,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    def render_chat_interface(self):
        """Render chat interface."""
        st.title("RAG Q&A Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show citations if available
                if message["role"] == "assistant" and "citations" in message:
                    with st.expander("Citations"):
                        for i, citation in enumerate(message["citations"]):
                            st.write(f"**Citation {i+1}:**")
                            st.write(f"Document: {citation.get('filename', 'Unknown')}")
                            st.write(f"Content: {citation.get('content', '')[:200]}...")
                            st.write(f"Similarity: {citation.get('similarity_score', 0):.3f}")
                
                # Show metadata if available
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    with st.expander("Response Metadata"):
                        st.write(f"**Grounding Score:** {metadata.get('grounding_score', 0):.3f}")
                        st.write(f"**Hallucination Detected:** {metadata.get('hallucination_detected', False)}")
                        st.write(f"**Tokens Used:** {metadata.get('tokens_used', 0)}")
                        st.write(f"**Estimated Cost:** ${metadata.get('estimated_cost', 0):.4f}")
                        st.write(f"**Processing Time:** {metadata.get('processing_time', 0):.3f}s")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            if st.session_state.rag_initialized:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Get settings
                            settings = self.render_settings()
                            
                            # Query RAG system
                            response = self.rag_core.query(
                                prompt,
                                top_k=settings["top_k"],
                                threshold=settings["threshold"]
                            )
                            
                            # Display response
                            st.markdown(response.answer)
                            
                            # Add assistant message with metadata
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response.answer,
                                "citations": response.citations,
                                "metadata": {
                                    "grounding_score": response.grounding_score,
                                    "hallucination_detected": response.hallucination_detected,
                                    "tokens_used": response.tokens_used,
                                    "estimated_cost": response.estimated_cost,
                                    "processing_time": response.processing_time
                                }
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please initialize the RAG system first.")
    
    def render_main_interface(self):
        """Render main interface."""
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Documents", "Evaluation", "Analytics"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_documents_tab()
        
        with tab3:
            self.render_evaluation_tab()
        
        with tab4:
            self.render_analytics_tab()
    
    def render_documents_tab(self):
        """Render documents tab."""
        st.header("Document Management")
        
        if not st.session_state.rag_initialized:
            st.info("Please initialize the RAG system first.")
            return
        
        try:
            # Get documents from database
            documents = self.db_manager.get_documents()
            
            if not documents:
                st.info("No documents uploaded yet.")
                return
            
            st.write(f"**Total Documents:** {len(documents)}")
            
            # Display documents
            for doc in documents:
                with st.expander(f"{doc['filename']} ({doc['file_type']})"):
                    st.write(f"**File Path:** {doc['file_path']}")
                    st.write(f"**Created:** {doc['created_at']}")
                    st.write(f"**Content Preview:** {doc['content'][:200]}...")
                    
                    # Show chunks
                    chunks = self.db_manager.get_chunks_by_document(doc['id'])
                    st.write(f"**Chunks:** {len(chunks)}")
                    
                    if st.checkbox(f"Show chunks for {doc['filename']}"):
                        for chunk in chunks:
                            st.write(f"**Chunk {chunk['chunk_index']}:** {chunk['content'][:100]}...")
        
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
    
    def render_evaluation_tab(self):
        """Render evaluation tab."""
        st.header("Evaluation")
        
        if not st.session_state.rag_initialized:
            st.info("Please initialize the RAG system first.")
            return
        
        # Evaluation controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Full Evaluation"):
                self.run_evaluation()
        
        with col2:
            if st.button("Load Evaluation Data"):
                eval_data = self.evaluator.load_eval_data()
                st.write(f"Loaded {len(eval_data)} evaluation items")
        
        # Evaluation results
        if st.checkbox("Show Evaluation Results"):
            try:
                # Load results from file
                import os
                if os.path.exists("eval_report.json"):
                    with open("eval_report.json", 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    st.subheader("Overall Metrics")
                    overall = results["overall_metrics"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Exact Match Rate", f"{overall['exact_match_rate']:.3f}")
                    with col2:
                        st.metric("Average F1 Score", f"{overall['average_f1_score']:.3f}")
                    with col3:
                        st.metric("Citation Accuracy", f"{overall['average_citation_accuracy']:.3f}")
                    with col4:
                        st.metric("Hallucination Rate", f"{overall['hallucination_rate']:.3f}")
                    
                    # Cost and performance
                    st.subheader("Cost and Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Cost", f"${overall['total_estimated_cost']:.4f}")
                    with col2:
                        st.metric("Total Tokens", f"{overall['total_tokens_used']:,}")
                    with col3:
                        st.metric("Avg Processing Time", f"{overall['average_processing_time']:.3f}s")
                
            except Exception as e:
                st.error(f"Error loading evaluation results: {str(e)}")
    
    def render_analytics_tab(self):
        """Render analytics tab."""
        st.header("Analytics")
        
        if not st.session_state.rag_initialized:
            st.info("Please initialize the RAG system first.")
            return
        
        try:
            # Cost analytics
            st.subheader("Cost Analytics")
            cost_summary = self.observability.get_cost_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${cost_summary['total_cost']:.4f}")
            with col2:
                st.metric("Total Queries", cost_summary['total_queries'])
            with col3:
                st.metric("Avg Cost/Query", f"${cost_summary['average_cost_per_query']:.4f}")
            
            # Performance analytics
            st.subheader("Performance Analytics")
            perf_summary = self.observability.get_performance_summary()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Operations", perf_summary['total_operations'])
            with col2:
                st.metric("Avg Duration", f"{perf_summary['average_duration_ms']:.2f}ms")
            
            # Recent queries
            st.subheader("Recent Queries")
            recent_queries = self.db_manager.get_recent_queries(limit=10)
            
            for query in recent_queries:
                with st.expander(f"Query {query['id']}: {query['query_text'][:50]}..."):
                    st.write(f"**Response:** {query['response'][:200]}...")
                    st.write(f"**Grounding Score:** {query['grounding_score']:.3f}")
                    st.write(f"**Hallucination:** {query['hallucination_detected']}")
                    st.write(f"**Cost:** ${query['estimated_cost']:.4f}")
                    st.write(f"**Time:** {query['processing_time']:.3f}s")
        
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def run(self):
        """Run the web UI."""
        st.set_page_config(
            page_title="RAG Q&A App",
            page_icon="🤖",
            layout="wide"
        )
        
        # Main interface
        self.render_main_interface()

def main():
    """Main function to run the web UI."""
    ui = RAGWebUI()
    ui.run()

if __name__ == "__main__":
    main()
