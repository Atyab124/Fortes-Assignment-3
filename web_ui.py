"""Streamlit web interface for the RAG Q&A system."""

import streamlit as st
import time
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import tempfile
import os

# Import our modules
from config import Config
from rag_core import RAGSystem
from guardrails import SafetyGuardrails
from attribution import AttributionAnalyzer
from observability import RAGMonitor, MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system (cached)."""
    try:
        config = Config()
        rag_system = RAGSystem(config)
        guardrails = SafetyGuardrails(config)
        attribution_analyzer = AttributionAnalyzer()
        metrics_collector = MetricsCollector()
        monitor = RAGMonitor(rag_system, metrics_collector)
        
        return rag_system, guardrails, attribution_analyzer, monitor, config
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None, None, None

def display_system_stats(rag_system: RAGSystem, monitor: RAGMonitor):
    """Display system statistics."""
    try:
        stats = rag_system.get_document_stats()
        system_metrics = monitor.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", stats['total_documents'])
        with col2:
            st.metric("Chunks", stats['total_chunks'])
        with col3:
            st.metric("Embeddings", stats['total_embeddings'])
        with col4:
            st.metric("Memory (MB)", f"{system_metrics.memory_usage_mb:.1f}")
        
        # File types breakdown
        if stats['file_types']:
            st.subheader("📁 File Types")
            for file_type, count in stats['file_types'].items():
                st.write(f"• {file_type}: {count} files")
                
    except Exception as e:
        st.error(f"Error loading stats: {e}")

def display_sources(sources: List[Dict[str, Any]]):
    """Display retrieved sources."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source['filename']} (Similarity: {source['similarity']:.3f})"):
            st.write("**Content:**")
            st.write(source['content'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Document ID:** {source.get('document_id', 'N/A')}")
            with col2:
                st.write(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")

def display_attribution_analysis(attribution_analyzer: AttributionAnalyzer, 
                               answer: str, sources: List[Dict[str, Any]]):
    """Display attribution analysis."""
    try:
        attributions = attribution_analyzer.analyze_response(answer, sources)
        report = attribution_analyzer.generate_attribution_report(attributions)
        
        st.subheader("🔍 Attribution Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Quality Score", f"{report['quality_score']:.3f}")
        with col2:
            st.metric("Hallucinated Sentences", f"{report['hallucinated_sentences']}/{report['total_sentences']}")
        with col3:
            st.metric("High Confidence", report['attribution_summary']['high_confidence'])
        
        # Attribution breakdown
        st.write("**Attribution Breakdown:**")
        attribution_data = {
            'High Confidence': report['attribution_summary']['high_confidence'],
            'Medium Confidence': report['attribution_summary']['medium_confidence'],
            'Low Confidence': report['attribution_summary']['low_confidence'],
            'No Attribution': report['attribution_summary']['no_attribution']
        }
        
        st.bar_chart(attribution_data)
        
        # Problematic sentences
        if report['problematic_sentences']:
            st.write("**⚠️ Problematic Sentences:**")
            for prob in report['problematic_sentences'][:3]:
                if prob['reason'] == 'hallucinated':
                    st.error(f"🚨 {prob['sentence'][:100]}...")
                else:
                    st.warning(f"⚠️ {prob['sentence'][:100]}...")
                    
    except Exception as e:
        st.error(f"Error in attribution analysis: {e}")

def handle_file_upload(rag_system: RAGSystem):
    """Handle file upload and ingestion."""
    st.subheader("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: .txt, .md, .pdf, .docx"
    )
    
    if uploaded_files:
        if st.button("Upload and Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Ingest the file
                    status_text.text(f"Processing {uploaded_file.name}...")
                    result = rag_system.ingest_document(tmp_path)
                    results.append(result)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    results.append({'filename': uploaded_file.name, 'success': False, 'error': str(e)})
            
            # Display results
            status_text.text("Processing complete!")
            
            successful = sum(1 for r in results if r.get('success', False))
            st.success(f"Successfully processed {successful}/{len(results)} files")
            
            # Show detailed results
            with st.expander("Detailed Results"):
                for result in results:
                    if result.get('success'):
                        st.success(f"✅ {result['filename']}: {result.get('chunks_created', 0)} chunks created")
                    else:
                        st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("A local, intelligent retrieval-augmented generation system")
    
    # Initialize system
    rag_system, guardrails, attribution_analyzer, monitor, config = initialize_system()
    
    if rag_system is None:
        st.error("Failed to initialize the system. Please check the logs.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.SIMILARITY_THRESHOLD,
            step=0.1,
            help="Minimum similarity score for retrieved chunks"
        )
        
        # Top-k results
        top_k = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=config.TOP_K,
            help="Number of top similar chunks to retrieve"
        )
        
        # System stats
        st.header("📊 System Status")
        display_system_stats(rag_system, monitor)
        
        # File upload
        st.header("📁 Document Management")
        handle_file_upload(rag_system)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Configuration", "📖 Help"])
    
    with tab1:
        st.header("Ask a Question")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])
                
                # Display sources if available
                if message.get("sources"):
                    display_sources(message["sources"])
                
                # Display attribution analysis if available
                if message.get("attribution_report") and attribution_analyzer:
                    with st.expander("🔍 Attribution Analysis"):
                        display_attribution_analysis(
                            attribution_analyzer,
                            message["content"],
                            message["sources"]
                        )
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Validate query
            is_valid, processed_query, validation_metadata = guardrails.validate_query(prompt)
            
            if not is_valid:
                st.error(f"❌ Query rejected: {processed_query}")
                st.info("The query was flagged as potentially unsafe or inappropriate.")
                return
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "avatar": "👤"
            })
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                sources_placeholder = st.empty()
                attribution_placeholder = st.empty()
                
                try:
                    # Query the system
                    with st.spinner("Thinking..."):
                        result = rag_system.query(
                            processed_query,
                            top_k=top_k,
                            similarity_threshold=similarity_threshold
                        )
                    
                    # Monitor the query
                    monitor.monitor_query(processed_query, result)
                    
                    # Display answer with streaming effect
                    answer = result['answer']
                    full_response = ""
                    for word in answer.split():
                        full_response += word + " "
                        time.sleep(0.02)  # Small delay for streaming effect
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Display sources
                    if result.get('sources'):
                        sources_placeholder.subheader("📚 Sources")
                        display_sources(result['sources'])
                    
                    # Display attribution analysis
                    if result.get('sources') and attribution_analyzer:
                        with attribution_placeholder.expander("🔍 Attribution Analysis"):
                            display_attribution_analysis(
                                attribution_analyzer,
                                result['answer'],
                                result['sources']
                            )
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "avatar": "🤖",
                        "sources": result.get('sources', []),
                        "attribution_report": True
                    })
                    
                    # Display response metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{result['query_time']:.2f}s")
                    with col2:
                        st.metric("Sources Retrieved", len(result.get('sources', [])))
                    with col3:
                        st.metric("Success", "✅" if result.get('retrieval_success') else "❌")
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    logger.error(f"Query error: {e}")
    
    with tab2:
        st.header("📊 Analytics Dashboard")
        
        try:
            # Get monitoring report
            report = monitor.generate_report(hours=24)
            
            # Query metrics
            st.subheader("📈 Query Metrics (Last 24 Hours)")
            query_metrics = report['query_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", query_metrics['total_queries'])
            with col2:
                st.metric("Success Rate", f"{query_metrics['success_rate']:.1%}")
            with col3:
                st.metric("Avg Response Time", f"{query_metrics['avg_response_time']:.2f}s")
            with col4:
                st.metric("Avg Similarity", f"{query_metrics['avg_max_similarity']:.3f}")
            
            # Response time distribution
            if query_metrics['response_time_distribution']:
                st.subheader("⏱️ Response Time Distribution")
                st.bar_chart(query_metrics['response_time_distribution'])
            
            # Recommendations
            if report['recommendations']:
                st.subheader("💡 Recommendations")
                for rec in report['recommendations']:
                    st.info(rec)
                    
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with tab3:
        st.header("⚙️ System Configuration")
        
        # Current configuration
        st.subheader("Current Settings")
        config_data = {
            "Embedding Model": config.EMBEDDING_MODEL,
            "Chat Model": config.CHAT_MODEL,
            "Chunk Size": config.CHUNK_SIZE,
            "Chunk Overlap": config.CHUNK_OVERLAP,
            "Similarity Threshold": config.SIMILARITY_THRESHOLD,
            "Top-K": config.TOP_K
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}:** {value}")
        
        # System actions
        st.subheader("System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Rebuild Index"):
                with st.spinner("Rebuilding index..."):
                    rag_system.rebuild_index()
                st.success("Index rebuilt successfully!")
        
        with col2:
            if st.button("🗑️ Clear All Data"):
                if st.checkbox("I understand this will delete all data"):
                    rag_system.clear_all_data()
                    st.success("All data cleared!")
    
    with tab4:
        st.header("📖 Help & Documentation")
        
        st.markdown("""
        ## How to Use the RAG Q&A System
        
        ### 1. Upload Documents
        - Use the sidebar to upload `.txt`, `.md`, `.pdf`, or `.docx` files
        - The system will automatically process and index your documents
        
        ### 2. Ask Questions
        - Type your questions in the chat interface
        - The system will retrieve relevant information and generate answers
        - Each answer includes source citations and attribution analysis
        
        ### 3. Understanding the Results
        - **Answer**: The generated response based on your documents
        - **Sources**: Retrieved document chunks with similarity scores
        - **Attribution Analysis**: Quality assessment and hallucination detection
        
        ### 4. Settings
        - **Similarity Threshold**: Minimum similarity for retrieved chunks
        - **Number of Sources**: How many chunks to retrieve
        - **System Stats**: Monitor your document collection
        
        ### Features
        - ✅ Local processing (no external API calls)
        - ✅ Safety guardrails (prompt injection detection, PII redaction)
        - ✅ Attribution and hallucination detection
        - ✅ Real-time monitoring and analytics
        - ✅ Support for multiple document formats
        
        ### Tips
        - Ask specific questions for better results
        - Check the similarity scores of sources
        - Review attribution analysis for answer quality
        - Use the analytics tab to monitor system performance
        """)

if __name__ == "__main__":
    main()
