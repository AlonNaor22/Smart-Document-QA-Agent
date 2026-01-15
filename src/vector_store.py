"""
Vector Store Module
===================

This module handles the "magic" of semantic search - finding text chunks
that are MEANINGFULLY similar to a question, not just keyword matches.

WHAT ARE EMBEDDINGS? (The Core Concept)
---------------------------------------

Embeddings convert text into numbers (vectors) that capture MEANING.

Traditional search: "dog" matches "dog" but NOT "puppy" or "canine"
Semantic search: "dog" matches "puppy", "canine", "pet", etc.

How? Each word/sentence becomes a point in a high-dimensional space:
- "dog" -> [0.2, -0.5, 0.8, 0.1, ...]  (384 numbers)
- "puppy" -> [0.19, -0.48, 0.79, 0.12, ...]  (very similar!)
- "car" -> [-0.7, 0.3, -0.2, 0.9, ...]  (very different)

Similar meanings = close points in this space.

WHAT IS A VECTOR DATABASE?
--------------------------

A regular database: Stores rows, searches by exact matches or patterns
A vector database: Stores vectors, searches by similarity (distance)

When you ask "What is the refund policy?", the system:
1. Converts your question to a vector
2. Finds stored vectors that are closest (most similar)
3. Returns the text chunks those vectors represent

This is MUCH more powerful than keyword search!

THE EMBEDDINGS MODEL WE USE
---------------------------

"all-MiniLM-L6-v2" from HuggingFace:
- Creates 384-dimensional vectors
- Runs locally on your CPU (no API costs!)
- Good balance of speed and quality
- Trained on over 1 billion sentence pairs

CHROMADB
--------

ChromaDB is our vector database. It:
- Stores vectors efficiently
- Performs fast similarity search
- Persists to disk (survives restarts)
- Is free and open-source

USAGE EXAMPLE
-------------
    from src.vector_store import create_vector_store, similarity_search

    # Create store from document chunks
    vector_store = create_vector_store(chunks)

    # Find relevant chunks for a question
    results = similarity_search(vector_store, "What is the main topic?")

    for doc in results:
        print(doc.page_content)
"""

import os
import shutil

# Use the new HuggingFace integration (fixes deprecation warning)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import centralized configuration
from src.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, TOP_K_RESULTS


def create_embeddings_model():
    """
    Create the embeddings model that converts text to vectors.

    This is the "translator" between human language and numbers.
    The model was trained on millions of text pairs to learn that
    similar sentences should have similar vectors.

    Returns:
        HuggingFaceEmbeddings: Model ready to convert text to vectors

    Technical Details:
        - Model: all-MiniLM-L6-v2
        - Output: 384-dimensional vectors
        - Speed: ~100 sentences/second on CPU
        - Memory: ~100MB model size

    Why These Settings:
        - device='cpu': Works everywhere (GPU would be faster but not required)
        - normalize_embeddings=True: Makes similarity comparison more accurate

    Example:
        >>> embeddings = create_embeddings_model()
        >>> vector = embeddings.embed_query("Hello world")
        >>> print(len(vector))  # 384 dimensions
        384
    """
    embeddings = HuggingFaceEmbeddings(
        # Model name from HuggingFace Hub
        # Other options: "all-mpnet-base-v2" (higher quality, slower)
        model_name=EMBEDDING_MODEL,

        # Run on CPU - works on any machine
        # Change to 'cuda' if you have an NVIDIA GPU for 10x speed
        model_kwargs={'device': 'cpu'},

        # Normalize vectors to unit length
        # This makes cosine similarity work better
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def clear_vector_store(persist_directory: str = CHROMA_PERSIST_DIR):
    """
    Delete the existing vector store to start fresh.

    This removes all previously stored embeddings so that when you
    load a new document, you only search within that document.

    Args:
        persist_directory: Folder where the database is stored

    Why This Exists:
        ChromaDB adds new documents to existing data by default.
        When loading a different document, we want to start fresh
        so searches only return results from the NEW document.
    """
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)


def create_vector_store(chunks: list, persist_directory: str = CHROMA_PERSIST_DIR):
    """
    Create a new vector store from document chunks.

    IMPORTANT: This clears any existing data first! Each time you load
    a new document, the old embeddings are deleted and replaced.

    This is where the "indexing" happens:
    1. Clear old database (if exists)
    2. Each chunk's text is converted to a vector
    3. Vectors are stored in ChromaDB
    4. The database is saved to disk

    Args:
        chunks: List of Document objects from document_loader
        persist_directory: Folder to save the database

    Returns:
        Chroma: Vector store ready for similarity search

    What Happens Inside:
        For each chunk:
        1. Extract the text (page_content)
        2. Run through embeddings model -> 384 numbers
        3. Store vector + original text + metadata in ChromaDB

    Time Complexity:
        - First run: Downloads model (~100MB) + embeds all chunks
        - Subsequent runs: Much faster (model cached)
        - Rough estimate: 1-2 seconds per 100 chunks

    Example:
        >>> chunks = load_and_split_pdf("document.pdf")
        >>> vector_store = create_vector_store(chunks)
        >>> print("Database created and saved!")

    Pipeline Position:
        load_pdf -> split -> [create_vector_store] -> search -> LLM
    """
    # Clear any existing database so we start fresh
    # This ensures searches only return results from the NEW document
    clear_vector_store(persist_directory)

    # Get the embeddings model
    embeddings = create_embeddings_model()

    # Create ChromaDB from documents
    # This does the heavy lifting:
    # - Embeds all chunks (can take 30+ seconds for large documents)
    # - Creates the vector index for fast search
    # - Saves everything to persist_directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store


def load_vector_store(persist_directory: str = CHROMA_PERSIST_DIR):
    """
    Load an existing vector store from disk.

    Use this when you've already processed a document and want to
    continue asking questions without re-embedding everything.

    Args:
        persist_directory: Folder where the database was saved

    Returns:
        Chroma: Vector store loaded from disk

    When to Use:
        - Restarting the application
        - Asking more questions about the same document
        - Saving API costs (don't re-embed unchanged documents)

    Example:
        >>> # First session: create and save
        >>> vector_store = create_vector_store(chunks)

        >>> # Later session: just load
        >>> vector_store = load_vector_store()
        >>> results = similarity_search(vector_store, "my question")

    Note:
        If the database doesn't exist, this will create an empty one.
        Always check if you need to process documents first.
    """
    # Get the same embeddings model (must match what was used to create)
    embeddings = create_embeddings_model()

    # Load the existing database
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store


def similarity_search(vector_store, query: str, k: int = TOP_K_RESULTS) -> list:
    """
    Find the most relevant chunks for a given question.

    This is the "retrieval" step - the R in RAG.

    How It Works:
        1. Your question is converted to a vector
        2. ChromaDB finds the k closest vectors (by cosine similarity)
        3. Returns the original text chunks for those vectors

    Args:
        vector_store: The Chroma vector store
        query: The user's question (natural language)
        k: Number of chunks to return (default from config)

    Returns:
        List of Document objects, ordered by relevance (most relevant first)

    Understanding Similarity:
        Cosine similarity measures the angle between vectors:
        - 1.0 = identical meaning
        - 0.0 = unrelated
        - -1.0 = opposite meaning (rare in practice)

    Example:
        >>> results = similarity_search(vector_store, "What is the deadline?")
        >>> print(len(results))  # Returns k chunks
        4
        >>> print(results[0].page_content)  # Most relevant chunk
        "The project deadline is December 31st..."

    Why k=4 by Default:
        - 1-2: Might miss relevant info spread across chunks
        - 3-5: Good balance of relevance and coverage
        - 6+: Starts including less relevant chunks, adds noise

    Pipeline Position:
        load -> split -> embed -> [similarity_search] -> LLM
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def similarity_search_with_scores(vector_store, query: str, k: int = TOP_K_RESULTS) -> list:
    """
    Find relevant chunks AND their similarity scores.

    Same as similarity_search, but also returns how similar each chunk is.
    Useful for debugging or showing confidence to users.

    Args:
        vector_store: The Chroma vector store
        query: The user's question
        k: Number of chunks to return

    Returns:
        List of tuples: [(Document, score), ...]
        Score is between 0 and 1 (higher = more similar)

    Example:
        >>> results = similarity_search_with_scores(vector_store, "refund policy")
        >>> for doc, score in results:
        ...     print(f"Score: {score:.2f} - {doc.page_content[:50]}...")
        Score: 0.89 - "Our refund policy allows returns within 30 days..."
        Score: 0.76 - "For refunds, please contact customer service..."
        Score: 0.54 - "Shipping and handling fees are non-refundable..."

    Use Cases:
        - Filter out low-confidence results (score < 0.5)
        - Show users how confident the system is
        - Debug why certain chunks are/aren't retrieved
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


# =============================================================================
# EDUCATIONAL NOTES FOR DEVELOPERS
# =============================================================================
"""
UNDERSTANDING VECTOR DIMENSIONS
-------------------------------

Why 384 dimensions? It's a balance:
- Too few (e.g., 50): Can't capture nuanced meanings
- Too many (e.g., 2048): Slower, more memory, diminishing returns
- 384: Sweet spot for general text

Each dimension represents some learned "feature" of language:
- Some might capture sentiment
- Some might capture topic
- Some might capture formality
- Most are abstract combinations

ALTERNATIVE EMBEDDING MODELS
----------------------------

1. OpenAI Embeddings (text-embedding-ada-002):
   - Higher quality, but costs money per token
   - 1536 dimensions
   - Requires API key

2. Cohere Embeddings:
   - Also costs money
   - Good for multilingual text

3. Other HuggingFace models:
   - all-mpnet-base-v2: Higher quality, 768 dimensions, slower
   - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
   - paraphrase-multilingual-MiniLM-L12-v2: For non-English text

ALTERNATIVE VECTOR DATABASES
----------------------------

1. FAISS (Facebook AI):
   - Very fast, great for large datasets
   - Doesn't persist to disk by default
   - More complex setup

2. Pinecone:
   - Cloud-hosted, scales infinitely
   - Costs money
   - No infrastructure to manage

3. Weaviate:
   - Open source, feature-rich
   - Can run locally or in cloud
   - Supports hybrid search (vector + keyword)

4. Milvus:
   - Open source, enterprise-grade
   - Great for very large scale
   - More complex setup

DEBUGGING TIPS
--------------

1. Check what's in your vector store:
   >>> collection = vector_store._collection
   >>> print(f"Total chunks: {collection.count()}")

2. See the raw similarity scores:
   >>> results = similarity_search_with_scores(vector_store, "my query")
   >>> for doc, score in results:
   ...     print(f"{score:.3f}: {doc.page_content[:50]}")

3. Test with known content:
   - If your document mentions "Python programming"
   - Search for "Python" - should find it
   - Search for "coding in Python" - should also find it (semantic!)
   - Search for "banana recipes" - should NOT find it

PERFORMANCE OPTIMIZATION
------------------------

For large document collections:
1. Batch embedding: Process chunks in batches of 32-64
2. Use GPU: Change device='cpu' to device='cuda'
3. Consider cloud vector DB for very large scale
"""
