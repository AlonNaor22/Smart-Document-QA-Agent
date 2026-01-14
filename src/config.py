"""
Configuration Settings for Smart Document Q&A Agent
====================================================

This file contains all configurable parameters for the application.
Centralizing settings here makes it easy to:
- Understand what can be customized
- Adjust behavior without modifying core code
- Experiment with different values

HOW TO USE:
-----------
Import settings in other files:
    from src.config import CHUNK_SIZE, MODEL_NAME

Then use them in your code:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)
"""

# =============================================================================
# DOCUMENT PROCESSING SETTINGS
# =============================================================================

# CHUNK_SIZE: Maximum number of characters per text chunk
#
# Why 1000?
# - Too small (e.g., 200): Loses context, AI gets fragmented information
# - Too large (e.g., 5000): May exceed token limits, less precise retrieval
# - 1000 chars ≈ 250 tokens ≈ 1-2 paragraphs (good balance)
#
# Adjust based on your documents:
# - Technical docs with code: Try 1500-2000 (code needs more context)
# - Simple text: 800-1000 works well
# - Legal documents: Try 1200-1500 (clauses need full context)
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: Characters shared between consecutive chunks
#
# Why overlap?
# - Prevents cutting sentences in the middle
# - Ensures context isn't lost at chunk boundaries
# - Example: "The contract expires on" | "December 31st"
#   With overlap: both chunks have "expires on December 31st"
#
# Rule of thumb: 10-20% of CHUNK_SIZE
CHUNK_OVERLAP = 200


# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================

# EMBEDDING_MODEL: The model that converts text to vectors (numbers)
#
# Options:
# - "all-MiniLM-L6-v2": Fast, lightweight, good for general text (384 dimensions)
# - "all-mpnet-base-v2": Higher quality, slower (768 dimensions)
# - "multi-qa-MiniLM-L6-cos-v1": Optimized for Q&A tasks
#
# These run locally on your CPU - no API costs!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================

# TOP_K_RESULTS: Number of text chunks to retrieve for each question
#
# Why 4?
# - Too few (1-2): Might miss relevant information
# - Too many (10+): Adds noise, increases API costs, may confuse the AI
# - 3-5 is usually optimal for most use cases
#
# Increase if:
# - Questions require information spread across the document
# - You're getting incomplete answers
#
# Decrease if:
# - Answers contain irrelevant information
# - You want faster, cheaper responses
TOP_K_RESULTS = 4


# =============================================================================
# LLM (LANGUAGE MODEL) SETTINGS
# =============================================================================

# MODEL_NAME: Which Claude model to use
#
# Options (as of 2024):
# - "claude-sonnet-4-5-20250929": Best balance of quality and speed (recommended)
# - "claude-haiku-4-5-20251001": Faster and cheaper, good for simple questions
# - "claude-opus-4-5-20250929": Highest quality, slower, more expensive
#
# For learning/testing: Use Haiku (cheapest)
# For production: Use Sonnet (best value)
MODEL_NAME = "claude-sonnet-4-5-20250929"

# TEMPERATURE: Controls randomness in AI responses
#
# Range: 0.0 to 1.0
# - 0.0: Deterministic, same input = same output (best for Q&A)
# - 0.5: Balanced creativity
# - 1.0: Maximum creativity/randomness
#
# For document Q&A, use 0.0 - we want factual, consistent answers
TEMPERATURE = 0.0

# MAX_TOKENS: Maximum length of AI response
#
# 1 token ≈ 4 characters ≈ 0.75 words
# - 500 tokens ≈ 375 words (short answer)
# - 1024 tokens ≈ 768 words (medium answer)
# - 2048 tokens ≈ 1536 words (long answer)
#
# Set based on expected answer length. Longer = more expensive.
MAX_TOKENS = 1024


# =============================================================================
# MEMORY SETTINGS
# =============================================================================

# MEMORY_SIZE: Number of previous Q&A exchanges to remember
#
# Why limit memory?
# - Each exchange adds to the prompt (costs money)
# - Too much history can confuse the AI
# - 5-10 exchanges is usually enough for follow-up questions
MEMORY_SIZE = 5


# =============================================================================
# STORAGE SETTINGS
# =============================================================================

# CHROMA_PERSIST_DIR: Where to save the vector database
#
# The database is saved to disk so you don't have to re-process
# documents every time you restart the application.
CHROMA_PERSIST_DIR = "chroma_db"


# =============================================================================
# API RETRY SETTINGS
# =============================================================================

# MAX_RETRIES: How many times to retry if API is overloaded
#
# Anthropic's API can sometimes be busy. Retrying with delays
# usually succeeds. 3 retries with exponential backoff works well.
MAX_RETRIES = 3
