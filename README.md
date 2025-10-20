# 🧠 RAG Q&A System

A fully local, Retrieval-Augmented Generation (RAG) system that answers questions over your own documents.  
Built with **Python**, **Ollama**, **FAISS**, **SQLite**, and **Streamlit**.

---

## 🌟 Overview

This project combines **structured storage (SQLite)** and **semantic search (FAISS)** with **local LLMs via Ollama**, delivering an intelligent and private document question-answering system.

### 🔹 Key Highlights

- **Local-Only AI** — Runs fully offline with Ollama; no external API keys required  
- **Multi-format Document Support** — Supports `.md`, `.txt`, `.pdf`, `.docx`  
- **Two-Layer Storage** —  
  - SQLite for document metadata and persistence  
  - FAISS for high-speed vector retrieval  
- **RAG Workflow** — Chunk, embed, index, retrieve, and answer grounded questions  
- **Safety & Guardrails** — Prompt-injection detection, PII redaction, grounding validation  
- **Attribution** — Each answer is linked back to its document sources  
- **Web & CLI Interfaces** — Streamlit app or command-line mode  

---

## ⚙️ System Architecture

```
         ┌──────────────────────────────┐
         │         User Query           │
         └─────────────┬────────────────┘
                       │
                       ▼
           [1] Query embedded via Ollama
                       │
                       ▼
         ┌────────────────────────────┐
         │        FAISS Index         │
         │ (fast vector similarity)   │
         └─────────────┬──────────────┘
                       │
          [2] Top-K similar chunks found
                       │
                       ▼
         ┌────────────────────────────┐
         │     SQLite Database        │
         │  (documents, chunks, meta) │
         └─────────────┬──────────────┘
                       │
        [3] Fetch metadata and content
                       │
                       ▼
         ┌────────────────────────────┐
         │  Ollama Chat Model (LLM)   │
         │ Generates grounded answer  │
         └────────────────────────────┘
```

### Storage Layers
| Layer | Role | Technology |
|--------|------|-------------|
| **SQLite** | Persistent structured storage (documents, chunks, metadata, and embeddings as blobs) | `sqlite3` |
| **FAISS** | In-memory index for fast semantic vector search | `faiss-cpu` / `faiss-gpu` |

The **VectorStoreManager** keeps both layers synchronized:
- Adds embeddings to FAISS and stores them in SQLite
- Enriches FAISS results with SQLite metadata
- Can rebuild FAISS index from SQLite if needed

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- [Install Ollama](https://ollama.ai)
- Pull required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5
  ```

### 2️⃣ Install & Run

```bash
git clone <repository-url>
cd rag-qa-system
pip install -r requirements.txt

# Start the Streamlit web interface
streamlit run web_ui.py

# or use the CLI
python main.py --interactive
```

---

## 💬 Using the System

### Web Interface
1. Upload one or more documents  
2. Ask natural language questions  
3. See grounded answers with source citations  
4. View system metrics in the Analytics tab  

### Command-Line Examples
```bash
# Ingest a folder of documents
python main.py --ingest ./sample_corpus/

# Ask a question
python main.py --query "What are the main AI ethics principles?"

# View system stats
python main.py --stats

# Clear data
python main.py --clear
```

### API Example
```python
from rag_core import RAGSystem
from config import Config

rag = RAGSystem(Config())
rag.ingest_document("ai_ethics.md")

result = rag.query("What are the main ethical challenges in AI?")
print(result["answer"])
```

---

## 🧩 Core Components

| Component | Description |
|------------|-------------|
| `document_processor.py` | Extracts text, chunks documents, and prepares metadata |
| `database.py` | Manages SQLite tables for documents, chunks, and embeddings |
| `vector_store.py` | Handles FAISS vector storage and retrieval |
| `rag_core.py` | Integrates all components: ingestion, querying, and answer generation |
| `guardrails.py` | Prevents unsafe or ungrounded responses |
| `attribution.py` | Traces model answers back to source chunks |
| `evaluation.py` | Measures retrieval and generation quality |
| `web_ui.py` | Streamlit front-end interface |
| `main.py` | CLI entrypoint |

---

## 🧠 How Retrieval Works

1. **Document Ingestion**
   - Text is extracted and split into overlapping chunks.
   - Each chunk is embedded using the Ollama embedding model.
   - Embeddings are stored in both SQLite (persistent) and FAISS (search index).

2. **Query**
   - Query is embedded using the same embedding model.
   - FAISS finds top-K most similar vectors.
   - `VectorStoreManager` maps FAISS IDs → chunk & document metadata from SQLite.
   - Relevant chunks are passed to the chat model (Qwen2.5) for grounded response generation.

3. **Answer Generation**
   - Model is prompted with system instructions to stay factual and cite context.
   - Returned answer includes source attribution and similarity scores.

---

## 🛡️ Safety & Attribution

| Feature | Description |
|----------|-------------|
| **Prompt Injection Detection** | Detects attempts to override system instructions |
| **PII Redaction** | Masks sensitive information before embedding |
| **Grounding Validation** | Ensures answers are backed by retrieved content |
| **Attribution Scoring** | Links every sentence of an answer to source chunks |

---

## 📊 Evaluation

Built-in evaluation harness using metrics such as:
- **Exact Match (EM)** — perfect answer matches  
- **F1 Score** — token-level overlap  
- **Semantic Similarity** — embedding cosine similarity  
- **Retrieval Success** — correct source retrieval rate  
- **Grounding & Attribution Scores**

Run evaluation:
```bash
python main.py --evaluate
```

---

## 🔍 Monitoring

`observability.py` provides real-time analytics:
- Query latency, similarity, and retrieval performance  
- Memory and CPU tracking  
- Long-term metrics reporting via `RAGMonitor`

```python
from observability import RAGMonitor
monitor = RAGMonitor(rag_system)
print(monitor.generate_report(hours=24))
```

---

## 🧪 Testing

```bash
pytest --cov=. --cov-report=html
```

Includes unit and integration tests for chunking, retrieval, guardrails, and evaluation.

---

## 📁 Project Structure

```
rag-qa-system/
├── config.py              # Configuration settings
├── database.py            # SQLite database operations
├── document_processor.py  # Text extraction and chunking
├── vector_store.py        # FAISS index management
├── rag_core.py            # Core RAG logic (ingestion + querying)
├── guardrails.py          # Safety mechanisms
├── attribution.py         # Source tracing and hallucination checks
├── evaluation.py          # Evaluation harness
├── observability.py       # Monitoring and reporting
├── main.py                # CLI entry point
├── web_ui.py              # Streamlit interface
├── eval.yaml              # Evaluation test cases
├── requirements.txt       # Dependencies
└── sample_corpus/         # Example documents
```

---

## 📄 License

Licensed under the MIT License.

---