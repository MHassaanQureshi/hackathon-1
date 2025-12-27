"""
Simple test to isolate the Qdrant client initialization issue
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"Qdrant URL: {qdrant_url}")
print(f"Qdrant API Key: {qdrant_api_key}")

# Test creating QdrantClient directly
try:
    from qdrant_client import QdrantClient

    print("Attempting to create QdrantClient with URL and API key...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    print("Successfully created QdrantClient!")

    # Test connection
    collections = client.get_collections()
    print(f"Successfully connected to Qdrant. Collections: {collections}")

except Exception as e:
    print(f"Error creating QdrantClient: {e}")
    import traceback
    traceback.print_exc()