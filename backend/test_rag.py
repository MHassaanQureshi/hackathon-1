"""
Test script for the RAG system
"""
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import RAGSystem

def test_rag_system():
    """Test the RAG system functionality"""
    print("Initializing RAG System...")
    rag_system = RAGSystem()

    print("\nTesting basic functionality:")

    # Test 1: Test embedding creation
    print("\n1. Testing embedding creation...")
    test_text = "This is a test sentence for embedding."
    try:
        embedding = rag_system._create_embedding(test_text)
        print(f"✓ Successfully created embedding with {len(embedding)} dimensions")
    except Exception as e:
        print(f"✗ Error creating embedding: {e}")
        return False

    # Test 2: Test text chunking
    print("\n2. Testing text chunking...")
    long_text = "This is a longer text that will be split into chunks. " * 20
    chunks = rag_system.chunk_text(long_text, chunk_size=100, overlap=20)
    print(f"✓ Successfully chunked text into {len(chunks)} chunks")

    # Test 3: Test indexing (using a simple test document)
    print("\n3. Testing content indexing...")
    test_content_id = "test_content_1"
    test_text = "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms."
    test_metadata = {
        "module": "Module 1",
        "chapter": "Chapter 1",
        "title": "ROS 2 Fundamentals"
    }

    try:
        success = rag_system.index_content(test_content_id, test_text, test_metadata)
        if success:
            print("✓ Successfully indexed test content")
        else:
            print("✗ Failed to index test content")
            return False
    except Exception as e:
        print(f"✗ Error indexing content: {e}")
        return False

    # Test 4: Test search functionality
    print("\n4. Testing search functionality...")
    try:
        search_results = rag_system.search_content("What is ROS 2?", limit=3)
        print(f"✓ Successfully searched content, found {len(search_results)} results")

        if search_results:
            print(f"  First result relevance score: {search_results[0]['relevance_score']:.3f}")
            print(f"  First result content preview: {search_results[0]['text'][:100]}...")
    except Exception as e:
        print(f"✗ Error searching content: {e}")
        return False

    # Test 5: Test conversation history
    print("\n5. Testing conversation history...")
    try:
        # Add a test conversation
        rag_system.add_to_conversation_history("test_session_1", "What is ROS 2?", "ROS 2 is a flexible framework for writing robot software.")

        # Get the history back
        history = rag_system.get_conversation_history("test_session_1")
        if len(history) > 0:
            print("✓ Successfully added and retrieved conversation history")
            print(f"  History contains {len(history)} interactions")
        else:
            print("✗ Conversation history is empty")
            return False
    except Exception as e:
        print(f"✗ Error with conversation history: {e}")
        return False

    # Test 6: Test response validation
    print("\n6. Testing response validation...")
    try:
        test_response = "ROS 2 is a flexible framework for writing robot software."
        test_chunks = [{"text": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software."}]
        is_valid = rag_system.validate_response(test_response, test_chunks)
        print(f"✓ Response validation test completed, is_valid: {is_valid}")

        # Test with invalid response
        invalid_response = "This response contains completely made up information not in the context."
        is_valid = rag_system.validate_response(invalid_response, test_chunks)
        print(f"  Validation for invalid response: {is_valid}")
    except Exception as e:
        print(f"✗ Error with response validation: {e}")
        return False

    # Test 7: Test full RAG chat flow
    print("\n7. Testing full RAG chat flow...")
    try:
        response = rag_system.chat_with_rag("What is ROS 2?", session_id="test_session_2")
        print(f"✓ Successfully completed RAG chat")
        print(f"  Response: {response['response'][:100]}...")
        print(f"  Sources: {len(response['sources'])} sources returned")
        print(f"  Session ID: {response['sessionId']}")
    except Exception as e:
        print(f"✗ Error with RAG chat: {e}")
        return False

    print("\n✓ All tests passed! RAG system is working correctly.")
    return True

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Run the tests
    success = test_rag_system()

    if success:
        print("\n🎉 RAG system implementation is successful!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)