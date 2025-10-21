#!/usr/bin/env python3
"""Test script to verify Ollama setup and model availability."""

import sys
from ollama import Client, list as ollama_list

def test_ollama_connection():
    """Test connection to Ollama server."""
    try:
        client = Client(host="http://localhost:11434")
        print("✅ Successfully connected to Ollama server")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Ollama server: {e}")
        print("Make sure Ollama is running: ollama serve")
        return None

def test_model_availability(client):
    """Test if required models are available."""
    if not client:
        return False
    
    try:
        # Get list of models
        models_response = client.list()
        print(f"Raw models response: {models_response}")
        
        # Extract model names
        if hasattr(models_response, 'models'):
            available_models = [model.model for model in models_response.models]
        elif isinstance(models_response, dict) and 'models' in models_response:
            available_models = [model.get('model', model.get('name', '')) for model in models_response['models']]
        else:
            print(f"Unexpected response format: {type(models_response)}")
            available_models = []
        
        print(f"Available models: {available_models}")
        
        # Check required models
        required_models = ["nomic-embed-text:latest", "qwen2.5:latest"]
        missing_models = []
        
        for model in required_models:
            if model in available_models:
                print(f"✅ Model '{model}' is available")
            else:
                print(f"❌ Model '{model}' is missing")
                missing_models.append(model)
        
        if missing_models:
            print(f"\nTo install missing models, run:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_embedding_model(client):
    """Test embedding model functionality."""
    if not client:
        return False
    
    try:
        from ollama import embed
        
        print("\n🧪 Testing embedding model...")
        response = embed(model="nomic-embed-text:latest", input="Hello, world!")
        
        if hasattr(response, 'embeddings'):
            embeddings = response.embeddings
        elif isinstance(response, dict) and 'embeddings' in response:
            embeddings = response['embeddings']
        else:
            embeddings = response
        
        if embeddings and len(embeddings) > 0:
            print(f"✅ Embedding generated successfully (dimension: {len(embeddings[0])})")
            return True
        else:
            print("❌ No embeddings generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        return False

def test_chat_model(client):
    """Test chat model functionality."""
    if not client:
        return False
    
    try:
        from ollama import chat
        
        print("\n🧪 Testing chat model...")
        response = chat(
            model="qwen2.5:latest",
            messages=[{'role': 'user', 'content': 'Hello! Please respond with just "Hi there!"'}]
        )
        
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
        elif isinstance(response, dict) and 'message' in response:
            content = response['message']['content']
        else:
            content = str(response)
        
        print(f"✅ Chat model response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chat model: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Ollama Setup for RAG Q&A System")
    print("=" * 50)
    
    # Test connection
    client = test_ollama_connection()
    
    # Test model availability
    if not test_model_availability(client):
        print("\n❌ Setup incomplete. Please install missing models.")
        sys.exit(1)
    
    # Test embedding model
    if not test_embedding_model(client):
        print("\n❌ Embedding model test failed.")
        sys.exit(1)
    
    # Test chat model
    if not test_chat_model(client):
        print("\n❌ Chat model test failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Ollama setup is complete.")
    print("You can now run the RAG Q&A system:")
    print("  streamlit run web_ui.py")
    print("  python main.py --interactive")

if __name__ == "__main__":
    main()
