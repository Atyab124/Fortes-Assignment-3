"""Main application entry point for the RAG Q&A system."""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from rag_core import RAGSystem
from guardrails import SafetyGuardrails
from attribution import AttributionAnalyzer
from observability import RAGMonitor, MetricsCollector
from evaluation import run_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_rag_system(config: Config) -> tuple:
    """Set up the complete RAG system with all components."""
    logger.info("Initializing RAG system...")
    
    # Core RAG system
    rag_system = RAGSystem(config)
    
    # Safety guardrails
    guardrails = SafetyGuardrails(config)
    
    # Attribution analyzer
    attribution_analyzer = AttributionAnalyzer()
    
    # Monitoring
    metrics_collector = MetricsCollector()
    monitor = RAGMonitor(rag_system, metrics_collector)
    
    logger.info("RAG system initialized successfully")
    return rag_system, guardrails, attribution_analyzer, monitor

def ingest_documents(rag_system: RAGSystem, path: str):
    """Ingest documents from a file or directory."""
    path = Path(path)
    
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return
    
    if path.is_file():
        logger.info(f"Ingesting single file: {path}")
        result = rag_system.ingest_document(str(path))
        if result['success']:
            logger.info(f"Successfully ingested {result['filename']}")
        else:
            logger.error(f"Failed to ingest {result['filename']}: {result.get('error')}")
    
    elif path.is_dir():
        logger.info(f"Ingesting directory: {path}")
        result = rag_system.ingest_directory(str(path))
        logger.info(f"Ingestion complete: {result['successful_files']}/{result['total_files']} files processed")
        
        if result['failed_files'] > 0:
            logger.warning(f"{result['failed_files']} files failed to process")
    
    else:
        logger.error(f"Invalid path: {path}")

def query_system(rag_system: RAGSystem, guardrails: SafetyGuardrails, 
                attribution_analyzer: AttributionAnalyzer, monitor: RAGMonitor):
    """Interactive query mode."""
    logger.info("Starting interactive query mode. Type 'quit' to exit.")
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Validate query through guardrails
            is_valid, processed_query, validation_metadata = guardrails.validate_query(query)
            
            if not is_valid:
                print(f"❌ Query rejected: {processed_query}")
                continue
            
            # Process query
            print("🔍 Processing your question...")
            result = rag_system.query(processed_query)
            
            # Monitor the query
            monitor.monitor_query(processed_query, result)
            
            # Display results
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(result['answer'])
            
            if result.get('sources'):
                print("\n📚 SOURCES:")
                print("-" * 40)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['filename']} (similarity: {source['similarity']:.3f})")
                    print(f"   {source['content'][:100]}...")
                    print()
            
            # Attribution analysis
            if result.get('sources') and attribution_analyzer:
                print("🔍 ATTRIBUTION ANALYSIS:")
                print("-" * 40)
                attributions = attribution_analyzer.analyze_response(result['answer'], result['sources'])
                report = attribution_analyzer.generate_attribution_report(attributions)
                
                print(f"Quality Score: {report['quality_score']:.3f}")
                print(f"Hallucinated Sentences: {report['hallucinated_sentences']}/{report['total_sentences']}")
                
                if report['problematic_sentences']:
                    print("\n⚠️  Problematic Sentences:")
                    for prob in report['problematic_sentences'][:3]:
                        print(f"  - {prob['sentence'][:80]}... ({prob['reason']})")
            
            print(f"\n⏱️  Response time: {result['query_time']:.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}")

def run_evaluation_mode(rag_system: RAGSystem, config: Config):
    """Run evaluation mode."""
    logger.info("Starting evaluation mode...")
    
    # Set up components for evaluation
    guardrails = SafetyGuardrails(config)
    attribution_analyzer = AttributionAnalyzer()
    
    # Run evaluation
    try:
        results = run_evaluation(
            rag_system=rag_system,
            eval_config_path=config.EVAL_CONFIG_PATH,
            output_path=config.EVAL_REPORT_PATH,
            guardrails=guardrails,
            attribution_analyzer=attribution_analyzer
        )
        
        logger.info("Evaluation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def show_stats(rag_system: RAGSystem, monitor: RAGMonitor):
    """Show system statistics."""
    stats = rag_system.get_document_stats()
    system_metrics = monitor.get_system_metrics()
    
    print("\n" + "="*50)
    print("SYSTEM STATISTICS")
    print("="*50)
    print(f"Documents: {stats['total_documents']}")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Embeddings: {stats['total_embeddings']}")
    print(f"File Types: {stats['file_types']}")
    print(f"Memory Usage: {system_metrics.memory_usage_mb:.1f} MB")
    print(f"CPU Usage: {system_metrics.cpu_usage_percent:.1f}%")
    print("="*50)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="RAG Q&A System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--ingest", help="Path to file or directory to ingest")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query mode")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--clear", action="store_true", help="Clear all data")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector index")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    try:
        # Set up RAG system
        rag_system, guardrails, attribution_analyzer, monitor = setup_rag_system(config)

        # --- Auto-ingest sample corpus if no --ingest argument ---
        if not args.ingest:
            sample_corpus_dir = config.get_sample_corpus_dir()
            if sample_corpus_dir.exists():
                try:
                    corpus_files = list(sample_corpus_dir.glob("*"))
                    if corpus_files:
                        logger.info(f"Auto-ingesting sample corpus from: {sample_corpus_dir}")
                        ingest_documents(rag_system, str(sample_corpus_dir))
                    else:
                        logger.warning(f"Sample corpus directory is empty: {sample_corpus_dir}")
                except Exception as e:
                    logger.error(f"Failed to ingest sample corpus automatically: {e}")
            else:
                logger.warning(f"Sample corpus directory not found: {sample_corpus_dir}")
        
        # Handle different modes
        if args.clear:
            rag_system.clear_all_data()
            print("✅ All data cleared")
            return
        
        if args.rebuild:
            rag_system.rebuild_index()
            print("✅ Vector index rebuilt")
            return
        
        if args.ingest:
            ingest_documents(rag_system, args.ingest)
            return
        
        if args.query:
            # Single query mode
            is_valid, processed_query, _ = guardrails.validate_query(args.query)
            if is_valid:
                result = rag_system.query(processed_query)
                monitor.monitor_query(processed_query, result)
                print(f"Answer: {result['answer']}")
            else:
                print(f"Query rejected: {processed_query}")
            return
        
        if args.evaluate:
            run_evaluation_mode(rag_system, config)
            return
        
        if args.stats:
            show_stats(rag_system, monitor)
            return
        
        if args.interactive:
            query_system(rag_system, guardrails, attribution_analyzer, monitor)
            return
        
        # Default: show help
        parser.print_help()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
