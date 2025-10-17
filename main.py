"""Main entry point for the RAG Q&A application."""

import argparse
import sys
from pathlib import Path

from rag_core import RAGCore
from evaluation import RAGEvaluator
from observability import ObservabilityManager
from database import DatabaseManager

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="RAG Q&A Application")
    parser.add_argument("--mode", choices=["web", "cli", "eval"], default="web", 
                       help="Application mode")
    parser.add_argument("--use-local", action="store_true", 
                       help="Use local model instead of API")
    parser.add_argument("--corpus-dir", type=str, default="sample_corpus",
                       help="Directory containing documents to process")
    parser.add_argument("--eval-file", type=str, default="eval.yaml",
                       help="Evaluation data file")
    parser.add_argument("--output", type=str, default="eval_report.json",
                       help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    if args.mode == "web":
        # Launch web interface
        try:
            import streamlit.web.cli as stcli
            import sys
            sys.argv = ["streamlit", "run", "web_ui.py"]
            stcli.main()
        except ImportError:
            print("Streamlit not installed. Please install with: pip install streamlit")
            sys.exit(1)
    
    elif args.mode == "cli":
        # Command-line interface
        run_cli(args)
    
    elif args.mode == "eval":
        # Evaluation mode
        run_evaluation(args)

def run_cli(args):
    """Run command-line interface."""
    print("Initializing RAG system...")
    
    # Initialize RAG system
    rag = RAGCore(use_local_model=args.use_local)
    
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.create_tables()
    
    # Initialize observability
    observability = ObservabilityManager()
    
    print("RAG system initialized successfully!")
    
    # Process corpus if directory exists
    corpus_path = Path(args.corpus_dir)
    if corpus_path.exists():
        print(f"Processing documents from {args.corpus_dir}...")
        
        # Get all supported files
        supported_extensions = ['.md', '.txt', '.pdf', '.docx']
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(list(corpus_path.glob(f"*{ext}")))
        
        if file_paths:
            results = rag.ingest_documents([str(f) for f in file_paths])
            print(f"Processed {results['documents_processed']} documents")
            print(f"Created {results['chunks_created']} chunks")
            
            if results['errors']:
                print(f"Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"  - {error}")
        else:
            print("No supported documents found in corpus directory")
    
    # Interactive CLI
    print("\nInteractive CLI - Type 'quit' to exit")
    print("Ask questions about your documents:")
    
    while True:
        try:
            question = input("\n> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("Thinking...")
            response = rag.query(question)
            
            print(f"\nAnswer: {response.answer}")
            
            if response.citations:
                print(f"\nCitations ({len(response.citations)}):")
                for i, citation in enumerate(response.citations, 1):
                    print(f"  {i}. {citation.get('filename', 'Unknown')} (score: {citation.get('similarity_score', 0):.3f})")
                    print(f"     {citation.get('content', '')[:100]}...")
            
            print(f"\nMetadata:")
            print(f"  - Grounding Score: {response.grounding_score:.3f}")
            print(f"  - Hallucination Detected: {response.hallucination_detected}")
            print(f"  - Tokens Used: {response.tokens_used}")
            print(f"  - Estimated Cost: ${response.estimated_cost:.4f}")
            print(f"  - Processing Time: {response.processing_time:.3f}s")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_evaluation(args):
    """Run evaluation mode."""
    print("Running evaluation...")
    
    # Initialize RAG system
    rag = RAGCore(use_local_model=args.use_local)
    
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.create_tables()
    
    # Process corpus if directory exists
    corpus_path = Path(args.corpus_dir)
    if corpus_path.exists():
        print(f"Processing documents from {args.corpus_dir}...")
        
        supported_extensions = ['.md', '.txt', '.pdf', '.docx']
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(list(corpus_path.glob(f"*{ext}")))
        
        if file_paths:
            results = rag.ingest_documents([str(f) for f in file_paths])
            print(f"Processed {results['documents_processed']} documents")
            print(f"Created {results['chunks_created']} chunks")
        else:
            print("No supported documents found in corpus directory")
            return
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    eval_data = evaluator.load_eval_data(args.eval_file)
    
    if not eval_data:
        print("No evaluation data found. Creating sample data...")
        evaluator._create_sample_eval_data()
        eval_data = evaluator.eval_data
    
    print(f"Running evaluation on {len(eval_data)} questions...")
    results = evaluator.run_evaluation(eval_data)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    output_file = evaluator.save_results(args.output)
    print(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()
