"""
Script to initialize the database with sample data for testing
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models import Base, BookModule, Chapter
from database import engine

def init_db():
    print("Initializing database...")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Check if we already have modules
        existing_modules = db.query(BookModule).count()
        if existing_modules > 0:
            print(f"Database already has {existing_modules} modules. Skipping sample data creation.")
            return

        # Create sample module
        sample_module = BookModule(
            title="Introduction to AI-Native Systems",
            description="An introduction to building AI-native applications and systems",
            order=1
        )
        db.add(sample_module)
        db.commit()
        db.refresh(sample_module)

        # Create sample chapter
        sample_chapter = Chapter(
            title="What is an AI-Native System?",
            module_id=sample_module.id,
            order=1,
            content="""An AI-native system is a software application that is designed from the ground up with artificial intelligence as a core component of its architecture. Unlike traditional applications that may have AI features bolted on, AI-native systems consider AI capabilities as fundamental to their design and functionality.

Key characteristics of AI-native systems include:

1. Intelligence by Design: AI is not an add-on but a core part of the system architecture
2. Adaptive Behavior: The system learns and adapts from interactions and data
3. Natural Interfaces: Interaction through natural language, voice, or other intuitive means
4. Continuous Learning: The system improves over time based on usage and feedback
5. Context Awareness: Understanding and responding to context in real-time

In this AI-Native Book system, the RAG (Retrieval Augmented Generation) functionality allows users to ask questions about the content and receive intelligent responses based on the indexed knowledge."""
        )
        db.add(sample_chapter)
        db.commit()

        print("Sample data created successfully:")
        print(f"- Module: {sample_module.title}")
        print(f"- Chapter: {sample_chapter.title}")

    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()