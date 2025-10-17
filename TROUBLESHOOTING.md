# Troubleshooting Guide

## Common Issues and Solutions

### 1. Ollama Connection Issues

**Error**: `Failed to initialize Ollama client: 'name'`

**Solution**:
1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Test your setup:
   ```bash
   python test_ollama_setup.py
   ```

3. Install required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5
   ```

### 2. Model Not Found Errors

**Error**: `Model not found` or `Model not available`

**Solution**:
1. Check available models:
   ```bash
   ollama list
   ```

2. Pull the required models:
   ```bash
   ollama pull nomic-embed-text  # For embeddings
   ollama pull qwen2.5           # For chat
   ```

3. Verify model installation:
   ```bash
   ollama show nomic-embed-text
   ollama show qwen2.5
   ```

### 3. Port Already in Use

**Error**: `Port 8501 is already in use`

**Solution**:
1. Find and kill the process using port 8501:
   ```bash
   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID_NUMBER> /F
   
   # Linux/Mac
   lsof -ti:8501 | xargs kill -9
   ```

2. Or use a different port:
   ```bash
   streamlit run web_ui.py --server.port 8502
   ```

### 4. Memory Issues

**Error**: `Out of memory` or slow performance

**Solution**:
1. Reduce batch size in `config.py`:
   ```python
   CHUNK_SIZE = 500  # Reduce from 1000
   TOP_K = 3         # Reduce from 5
   ```

2. Process fewer documents at once
3. Increase system RAM or use a smaller model

### 5. File Upload Issues

**Error**: `Unsupported file format`

**Solution**:
1. Check file extension is supported:
   - `.txt` - Plain text files
   - `.md` - Markdown files
   - `.pdf` - PDF documents
   - `.docx` - Word documents

2. Ensure file is not corrupted
3. Try converting to a supported format

### 6. Database Issues

**Error**: `Database locked` or `SQLite errors`

**Solution**:
1. Close any other instances of the application
2. Delete the database file and restart:
   ```bash
   rm vector_store.db
   python main.py --interactive
   ```

3. Rebuild the index:
   ```bash
   python main.py --rebuild
   ```

### 7. FAISS Index Issues

**Error**: `FAISS index corruption` or `Index not found`

**Solution**:
1. Clear and rebuild the index:
   ```bash
   python main.py --clear
   python main.py --rebuild
   ```

2. Check disk space availability
3. Verify FAISS installation:
   ```bash
   pip install faiss-cpu --upgrade
   ```

### 8. Import Errors

**Error**: `ModuleNotFoundError` or import issues

**Solution**:
1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Use a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

### 9. Performance Issues

**Slow response times or high memory usage**

**Solution**:
1. Optimize configuration:
   ```python
   # In config.py
   CHUNK_SIZE = 500
   CHUNK_OVERLAP = 100
   TOP_K = 3
   SIMILARITY_THRESHOLD = 0.8
   ```

2. Use smaller models if available
3. Process documents in smaller batches
4. Monitor system resources

### 10. Streamlit Issues

**Error**: Streamlit not loading or UI problems

**Solution**:
1. Update Streamlit:
   ```bash
   pip install streamlit --upgrade
   ```

2. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```

3. Try different browser or incognito mode
4. Check firewall settings

## Getting Help

### 1. Check Logs
Look at the console output for detailed error messages and stack traces.

### 2. Enable Debug Mode
Add debug logging to see more details:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. Test Individual Components
```bash
# Test document processing
python -c "from document_processor import DocumentProcessor; print('OK')"

# Test vector store
python -c "from vector_store import FAISSVectorStore; print('OK')"

# Test Ollama connection
python test_ollama_setup.py
```

### 4. System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- 2GB+ free disk space
- Ollama installed and running

### 5. Common Commands
```bash
# Start fresh
python main.py --clear
python main.py --ingest sample_corpus/

# Test everything
python test_ollama_setup.py
pytest

# Run evaluation
python main.py --evaluate

# Check system stats
python main.py --stats
```

## Still Having Issues?

1. **Check the GitHub issues** for similar problems
2. **Run the test suite** to identify specific failures:
   ```bash
   pytest -v
   ```
3. **Create a minimal reproduction** of the issue
4. **Include system information**:
   - Operating system
   - Python version
   - Ollama version
   - Error messages and logs
