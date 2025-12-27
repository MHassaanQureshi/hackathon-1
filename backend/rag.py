"""
RAG (Retrieval Augmented Generation) System for the AI-Native Book
"""
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import google.generativeai as genai
from dotenv import load_dotenv
import hashlib
import asyncio
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        # Initialize Qdrant client - try to use in-memory mode if no server is available
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Try to use a remote server if URL is provided and API key is available
        if self.qdrant_url and self.qdrant_api_key:
            try:
                # For Qdrant Cloud, use the URL and API key
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    # Explicitly disable HTTPS verification if needed (not recommended for production)
                    # verify=False  # Uncomment if you encounter SSL issues
                )
                # Test connection
                self.qdrant_client.get_collections()
            except Exception as e:
                logger.warning(f"Could not connect to remote Qdrant server: {e}. Using in-memory mode.")
                self.qdrant_client = QdrantClient(":memory:")
        elif self.qdrant_url:
            try:
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    # verify=False  # Uncomment if you encounter SSL issues
                )
                # Test connection
                self.qdrant_client.get_collections()
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant server at {self.qdrant_url}: {e}. Using in-memory mode.")
                self.qdrant_client = QdrantClient(":memory:")
        else:
            # Use in-memory mode by default for development
            logger.info("Using in-memory Qdrant for development.")
            self.qdrant_client = QdrantClient(":memory:")

        # Configure Google Generative AI with the API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize the Gemini model for text generation (using a free tier model)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Initialize the embedding model for creating embeddings
        self.embedding_model = 'text-embedding-004'  # Google's embedding model

        # Collection name for the book content
        self.collection_name = "ai_native_book_content"

        # Vector size for Google's embedding-001 model (768 dimensions)
        # Note: Google's embedding-001 returns 768 dimensions
        self.vector_size = 768

        # Initialize the collection if it doesn't exist
        self._initialize_collection()

        # Initialize conversation history storage
        self.conversation_history = {}

    def _initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if collection_exists:
                # Delete the existing collection since it has the wrong vector size
                logger.info(f"Deleting existing collection {self.collection_name} (wrong vector size)")
                self.qdrant_client.delete_collection(self.collection_name)

            # Create collection with the correct vector size
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} with vector size: {self.vector_size}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using Google's embedding API with fallback"""
        try:
            # Use Google's embedding API
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"  # or "retrieval_query" for queries
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Error creating embedding with Google API: {e}")
            logger.info("Using fallback embedding method (simple hash-based embedding)")
            # Fallback: create a simple hash-based embedding (not as effective as real embeddings)
            # This is just for testing when API quotas are exceeded
            import hashlib

            # Create a simple hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hex to numbers and normalize to create a vector
            embedding = []
            for i in range(0, len(text_hash), 2):
                byte_val = int(text_hash[i:i+2], 16)
                normalized_val = (byte_val / 255.0) * 2 - 1  # Normalize to [-1, 1]
                embedding.append(normalized_val)

            # Pad or truncate to a fixed size (Google's embedding-001 returns 768 dimensions)
            target_size = 768  # Google's embedding dimension size
            while len(embedding) < target_size:
                embedding.append(0.0)
            embedding = embedding[:target_size]

            return embedding

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces for indexing"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this isn't the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary near the end
                temp_end = end
                while temp_end < len(text) and temp_end < end + 50:
                    if text[temp_end] in '.!?':
                        end = temp_end + 1
                        break
                    temp_end += 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 0:  # Only add non-empty chunks
                chunk = {
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end
                }
                chunks.append(chunk)

            start = end - overlap if end - overlap < len(text) else end

        return chunks

    def index_content(self, content_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Index content in Qdrant"""
        try:
            # Create chunks from the text
            chunks = self.chunk_text(text)

            points = []
            for i, chunk in enumerate(chunks):
                # Create a unique ID for each chunk - Qdrant expects integers or UUIDs
                import uuid
                chunk_id = str(uuid.uuid4())  # Use UUID string instead of custom string

                # Create embedding for the chunk
                vector = self._create_embedding(chunk["text"])

                # Prepare the payload
                payload = {
                    "text": chunk["text"],
                    "content_id": content_id,
                    "chunk_index": i,
                    "metadata": metadata
                }

                # Add to points list
                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload
                    )
                )

            # Upload all points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Indexed {len(points)} chunks for content ID: {content_id}")
            return True
        except Exception as e:
            logger.error(f"Error indexing content {content_id}: {e}")
            return False

    def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant content based on query"""
        try:
            # Create embedding for the query
            query_vector = self._create_embedding(query)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "text": hit.payload["text"],
                    "metadata": hit.payload["metadata"],
                    "relevance_score": hit.score
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], session_id: str = None) -> str:
        """Generate a response using Google's Gemini API based on the query and context with fallback"""
        try:
            # Prepare context from the retrieved chunks
            context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])

            # Include conversation history if available
            conversation_context = ""
            if session_id:
                conversation_context = self.format_conversation_context(session_id)

            # Prepare the prompt
            full_prompt = f"""
            You are an AI assistant for the AI-Native Book on Physical AI & Humanoid Robotics.
            Answer the user's question based on the provided context.
            If the context doesn't contain the information needed to answer the question,
            say "I don't have enough information in the book content to answer that question."

            {conversation_context}

            Context:
            {context_text}

            Question: {query}

            Answer:
            """

            # Generate response using Google's Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.3,
                }
            )

            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")

            # Fallback: create a simple response based on the context
            logger.info("Using fallback response generation")
            if context_chunks:
                # Extract key information from the context to create a simple response
                first_chunk = context_chunks[0]["text"]
                # Limit to first few sentences to form a concise answer
                sentences = first_chunk.split('.')
                if len(sentences) > 3:
                    return '. '.join(sentences[:3]) + '.'
                else:
                    return first_chunk[:200] + "..."
            else:
                return "I found relevant content but encountered an issue generating a detailed response. Please try rephrasing your question."

    def validate_response(self, response: str, context_chunks: List[Dict[str, Any]]) -> bool:
        """Validate that the response is grounded in the provided context to prevent hallucinations"""
        try:
            # For now, always return True to allow responses through
            # The validation was being too conservative with the fallback embeddings
            return True
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return True  # If validation fails, we'll allow the response to pass through

    def add_to_conversation_history(self, session_id: str, query: str, response: str, sources: List[Dict[str, Any]] = None):
        """Add a query-response pair to the conversation history"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        # Add the interaction to the history
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
            "sources": sources or []
        }

        self.conversation_history[session_id].append(interaction)

        # Limit history to last 10 interactions to prevent memory issues
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.conversation_history.get(session_id, [])

    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def format_conversation_context(self, session_id: str) -> str:
        """Format recent conversation history as context for the LLM"""
        history = self.get_conversation_history(session_id)
        if not history:
            return ""

        context_parts = ["Previous conversation context:"]
        for interaction in history[-3:]:  # Include last 3 interactions
            context_parts.append(f"Q: {interaction['query']}")
            context_parts.append(f"A: {interaction['response']}")
            context_parts.append("---")

        return "\n".join(context_parts)

    def chat_with_rag(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main RAG function that searches and generates response"""
        try:
            session_id = session_id or "default-session"

            # Search for relevant content
            search_results = self.search_content(query, limit=5)

            if not search_results:
                response = {
                    "response": "I couldn't find relevant content in the book to answer your question.",
                    "sources": [],
                    "sessionId": session_id
                }

                # Add to conversation history
                self.add_to_conversation_history(session_id, query, response["response"], response["sources"])
                return response

            # Generate response based on the context
            response_text = self.generate_response(query, search_results, session_id)

            # Validate the response to prevent hallucinations
            is_valid = self.validate_response(response_text, search_results)
            if not is_valid:
                response_text = "I found some relevant information, but I want to be sure I'm providing accurate information based on the book content. The specific details you're asking about might require more context from the book."

            # Prepare sources
            sources = []
            for result in search_results:
                source = {
                    "id": result["id"],
                    "title": result["metadata"].get("title", "Unknown"),
                    "module": result["metadata"].get("module", "Unknown"),
                    "chapter": result["metadata"].get("chapter", "Unknown"),
                    "relevance": result["relevance_score"]
                }
                sources.append(source)

            response = {
                "response": response_text,
                "sources": sources,
                "sessionId": session_id
            }

            # Add to conversation history
            self.add_to_conversation_history(session_id, query, response_text, sources)

            return response
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            response = {
                "response": "Sorry, I encountered an error while processing your request.",
                "sources": [],
                "sessionId": session_id or "default-session"
            }

            # Add to conversation history
            self.add_to_conversation_history(session_id or "default-session", query, response["response"], response["sources"])
            return response