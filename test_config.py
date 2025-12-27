"""
Test script to verify the RAG system configuration with new Qdrant and Neon database URLs
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

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Database URL: {database_url}")

    if not qdrant_url:
        print("[ERROR] QDRANT_URL not found in environment")
        return False

    if not database_url:
        print("[ERROR] DATABASE_URL not found in environment")
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

        return True

    except Exception as e:
        print(f"[ERROR] Error initializing RAG system: {e}")
        return False

def test_database_connection():
    print("\nTesting database connection...")

    # Import database separately to avoid loading all env vars in Settings
    import importlib.util
    spec = importlib.util.spec_from_file_location("database", os.path.join(os.path.dirname(__file__), 'backend', 'database.py'))
    database = importlib.util.module_from_spec(spec)

    # Temporarily set only the database URL for this test
    original_db_url = os.environ.get('DATABASE_URL')

    try:
        # Create a temporary settings object with only database_url
        from pydantic_settings import BaseSettings

        class TempSettings(BaseSettings):
            database_url: str

            class Config:
                env_file = ".env"
                # Only allow the database_url field
                extra = "ignore"  # This will ignore other env vars

        # Load settings with only the database URL
        temp_settings = TempSettings(database_url=original_db_url)
        print(f"[SUCCESS] Database settings loaded: {temp_settings.database_url[:50]}...")

        # Now test the actual database connection
        from sqlalchemy import create_engine
        engine = create_engine(temp_settings.database_url)

        conn = engine.connect()
        print("[SUCCESS] Database connection successful")
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting configuration test...")

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    rag_success = test_rag_system()
    db_success = test_database_connection()

    if rag_success and db_success:
        print("\n[SUCCESS] All configuration tests passed!")
    else:
        print("\n[ERROR] Some configuration tests failed!")
        sys.exit(1)