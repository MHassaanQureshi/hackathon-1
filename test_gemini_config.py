"""
Test script to verify the RAG system configuration with new Qdrant, Neon database, and Gemini API
"""
import os
import sys
import logging

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from rag import RAGSystem

def test_rag_system():
    print("Testing RAG system configuration...")

    # Check environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    database_url = os.getenv("DATABASE_URL")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Database URL: {database_url}")
    print(f"Gemini API Key present: {gemini_api_key is not None and len(gemini_api_key) > 0}")

    if not qdrant_url:
        print("[ERROR] QDRANT_URL not found in environment")
        return False

    if not database_url:
        print("[ERROR] DATABASE_URL not found in environment")
        return False

    if not gemini_api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment")
        return False

    print("[SUCCESS] Environment variables are set")

    # Test RAG system initialization
    try:
        rag_system = RAGSystem()
        print("[SUCCESS] RAG system initialized successfully")

        # Test that it's using the configured Qdrant URL
        if rag_system.qdrant_url:
            print(f"[SUCCESS] RAG system is configured to use Qdrant at: {rag_system.qdrant_url}")
        else:
            print("[WARNING] RAG system is using in-memory mode")

        # Test that the model is set up
        if hasattr(rag_system, 'model'):
            print("[SUCCESS] Gemini model is configured")
        else:
            print("[ERROR] Gemini model is not configured")

        return True

    except Exception as e:
        print(f"[ERROR] Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_creation():
    print("\nTesting embedding creation...")
    try:
        rag_system = RAGSystem()
        test_text = "This is a test sentence for embedding."
        embedding = rag_system._create_embedding(test_text)

        if isinstance(embedding, list) and len(embedding) > 0:
            print(f"[SUCCESS] Embedding created successfully, length: {len(embedding)}")
            return True
        else:
            print("[ERROR] Embedding creation failed")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating embedding: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_response_generation():
    print("\nTesting response generation...")
    try:
        rag_system = RAGSystem()
        test_query = "What is the purpose of this AI system?"
        test_context = [{"text": "This is a test context for the RAG system."}]

        response = rag_system.generate_response(test_query, test_context)

        if response and len(response) > 0:
            print(f"[SUCCESS] Response generated successfully, length: {len(response)}")
            print(f"Response preview: {response[:100]}...")
            return True
        else:
            print("[ERROR] Response generation failed")
            return False
    except Exception as e:
        print(f"[ERROR] Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive configuration test...")

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    rag_success = test_rag_system()
    embedding_success = test_embedding_creation()
    response_success = test_response_generation()

    if rag_success and embedding_success and response_success:
        print("\n[SUCCESS] All configuration tests passed!")
    else:
        print("\n[ERROR] Some configuration tests failed!")
        sys.exit(1)