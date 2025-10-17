# RAG Q&A Application

A comprehensive Retrieval-Augmented Generation (RAG) system that answers questions over a local corpus of documents with advanced security guardrails, attribution, and evaluation capabilities.

## Features

### Core Functionality
- **Document Ingestion**: Process and chunk documents (MD, TXT, PDF, DOCX)
- **Vector Search**: FAISS-based similarity search with configurable thresholds
- **Text Generation**: Local or API-based response generation
- **Citation Alignment**: Automatic citation generation with source attribution
- **Hallucination Detection**: Identify unsupported claims in responses

### Security & Guardrails
- **Prompt Injection Detection**: Block malicious prompt injection attempts
- **PII Redaction**: Automatically detect and redact personal information
- **Query Validation**: Comprehensive security validation for all inputs
- **Response Sanitization**: Clean outputs for sensitive information

### Evaluation & Metrics
- **Comprehensive Evaluation**: Exact Match, F1 Score, Similarity metrics
- **Citation Accuracy**: Measure quality of source attribution
- **Cost Tracking**: Monitor token usage and estimated costs
- **Performance Monitoring**: Track processing times and system metrics

### User Interface
- **Streamlit Web UI**: Interactive chat interface with streaming responses
- **Document Management**: Upload and manage document corpus
- **Analytics Dashboard**: View system statistics and performance metrics
- **Evaluation Interface**: Run and analyze evaluation results

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd rag-qa-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p vector_db
mkdir -p logs
```

## Quick Start

### 1. Initialize the System
```python
from rag_core import RAGCore

# Initialize RAG system (use local model for demo)
rag = RAGCore(use_local_model=True)
```

### 2. Ingest Documents
```python
# Process documents from a directory
file_paths = ["sample_corpus/machine_learning_basics.md", "sample_corpus/neural_networks.txt"]
results = rag.ingest_documents(file_paths)
print(f"Processed {results['documents_processed']} documents, created {results['chunks_created']} chunks")
```

### 3. Query the System
```python
# Ask questions
response = rag.query("What is machine learning?")
print(f"Answer: {response.answer}")
print(f"Citations: {len(response.citations)}")
print(f"Grounding Score: {response.grounding_score}")
```

### 4. Run Evaluation
```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator(rag)
eval_data = evaluator.load_eval_data()
results = evaluator.run_evaluation(eval_data)
evaluator.print_summary()
```

## Web Interface

Launch the Streamlit web interface:
```bash
streamlit run web_ui.py
```

The web interface provides:
- Interactive chat with the RAG system
- Document upload and management
- Real-time evaluation and analytics
- System statistics and monitoring

## Configuration

Edit `config.py` to customize:
- Model settings (embedding model, generation model)
- Chunk size and overlap
- Similarity thresholds
- Security settings
- Logging and cost tracking

## Testing

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest

# Run specific test modules
python test_chunker.py
python test_retriever.py
python test_guardrails.py
python test_eval_math.py
```

## Evaluation

### Running Evaluation
1. Ensure you have documents in the `sample_corpus/` directory
2. Run the evaluation:
```python
from evaluation import RAGEvaluator
from rag_core import RAGCore

rag = RAGCore(use_local_model=True)
evaluator = RAGEvaluator(rag)

# Load evaluation data
eval_data = evaluator.load_eval_data()

# Run evaluation
results = evaluator.run_evaluation(eval_data)

# Print summary
evaluator.print_summary()

# Save results
evaluator.save_results()
```

### Evaluation Metrics
- **Exact Match (EM)**: Perfect match between expected and actual answers
- **F1 Score**: Harmonic mean of precision and recall
- **Similarity Score**: Text similarity using sequence matching
- **Citation Accuracy**: Quality of source attribution
- **Hallucination Detection**: Identification of unsupported claims
- **Grounding Score**: Confidence in retrieved information

## Sample Corpus

The `sample_corpus/` directory contains example documents:
- `machine_learning_basics.md`: Introduction to machine learning concepts
- `neural_networks.txt`: Comprehensive guide to neural networks
- `ai_ethics.md`: AI ethics principles and challenges

## Security Features

### Prompt Injection Detection
The system detects and blocks various types of prompt injection attempts:
- Instruction override attempts
- Role-playing prompts
- System prompt extraction
- Jailbreak attempts
- Social engineering

### PII Redaction
Automatically detects and redacts:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- URLs
- Dates of birth

### Query Validation
Comprehensive validation includes:
- Length limits (max 1000 characters)
- Empty query detection
- Injection pattern detection
- PII detection
- Malicious content filtering

## Architecture

### Core Components
- **DocumentProcessor**: Handles document ingestion and chunking
- **VectorStore**: FAISS-based vector similarity search
- **RAGCore**: Main RAG system with retrieval and generation
- **Guardrails**: Security and safety mechanisms
- **ObservabilityManager**: Logging, cost tracking, and monitoring
- **RAGEvaluator**: Comprehensive evaluation framework

### Data Flow
1. **Ingestion**: Documents → Chunking → Embeddings → Vector Store
2. **Query**: User Question → Security Validation → Retrieval → Generation
3. **Response**: Answer + Citations + Metadata → Security Sanitization
4. **Evaluation**: Test Cases → Metrics Calculation → Performance Analysis

## API Usage

### Basic RAG Operations
```python
from rag_core import RAGCore

# Initialize
rag = RAGCore(use_local_model=True)

# Ingest documents
rag.ingest_documents(["doc1.md", "doc2.txt"])

# Query system
response = rag.query("What is the main topic?")
print(response.answer)
print(response.citations)
```

### Security Operations
```python
from guardrails import Guardrails

guardrails = Guardrails()

# Validate query
validation = guardrails.validate_query("What is machine learning?")
if validation["is_safe"]:
    # Process query
    pass
else:
    print(f"Query blocked: {validation['block_reason']}")
```

### Evaluation Operations
```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator(rag)
results = evaluator.run_evaluation()
print(f"Exact Match Rate: {results['overall_metrics']['exact_match_rate']}")
```

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure all dependencies are installed
2. **Vector Store Issues**: Check FAISS installation and permissions
3. **Document Processing**: Verify file formats and encoding
4. **Memory Issues**: Reduce chunk size or use smaller models

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Streamlit, and FAISS
- Uses sentence-transformers for embeddings
- Implements comprehensive security guardrails
- Includes extensive evaluation framework

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases for examples
3. Open an issue on GitHub
4. Check the documentation and examples

---

**Note**: This is a demonstration RAG system. For production use, consider additional security measures, performance optimization, and scalability improvements.
