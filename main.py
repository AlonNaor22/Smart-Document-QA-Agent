#!/usr/bin/env python3
"""
Smart Document Q&A Agent - Main Entry Point
============================================

This is the command-line interface (CLI) for the Document Q&A system.
Run this file to start an interactive session where you can:
1. Load a PDF document
2. Ask questions about its content
3. Get AI-powered answers with source references

HOW TO RUN
----------
From the project directory:
    python main.py

Or if using a virtual environment:
    venv/Scripts/python main.py  (Windows)
    venv/bin/python main.py      (Mac/Linux)

WHAT HAPPENS WHEN YOU RUN THIS
------------------------------

1. Application starts and shows welcome message
2. You enter a path to a PDF file
3. System processes the document:
   - Extracts text from PDF
   - Splits into chunks
   - Creates embeddings (first time takes ~30 seconds)
   - Stores in vector database
4. You enter a chat loop where you can ask questions
5. For each question:
   - Relevant chunks are retrieved
   - Claude generates an answer
   - Sources are displayed
6. Conversation memory allows follow-up questions

COMMANDS
--------
- Type any question to ask about the document
- Type 'quit' or 'exit' to leave the application
- Type 'new' to load a different document
- Type 'clear' to reset conversation memory

PROJECT STRUCTURE REFRESHER
---------------------------
main.py (this file)
    |
    ├── src/document_loader.py  --> Loads and splits PDFs
    |
    ├── src/vector_store.py     --> Embeddings and search
    |
    ├── src/qa_chain.py         --> Q&A logic with Claude
    |
    └── src/config.py           --> All settings in one place
"""

import os
import sys

# Import our custom modules
# Each module handles a specific part of the RAG pipeline
from src.document_loader import load_and_split_pdf
from src.vector_store import create_vector_store, CHROMA_PERSIST_DIR
from src.qa_chain import create_qa_chain, ask_question, format_response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_screen():
    """
    Clear the terminal screen for a cleaner UI.

    Uses 'cls' on Windows, 'clear' on Mac/Linux.
    os.name == 'nt' checks if we're on Windows (NT kernel).
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """
    Display the application header/welcome message.

    This creates a nice visual separator and tells users
    what they can do with the application.
    """
    print("=" * 60)
    print("       Smart Document Q&A Agent")
    print("       Ask questions about your PDF documents")
    print("=" * 60)
    print()
    print("This application uses RAG (Retrieval Augmented Generation)")
    print("to answer questions based on YOUR documents, not general")
    print("knowledge. Answers are grounded in the actual content.")
    print()


def print_help():
    """
    Display available commands to the user.
    """
    print("\nAvailable commands:")
    print("  [any question]  - Ask about the document")
    print("  'quit' or 'exit'- Leave the application")
    print("  'new'           - Load a different document")
    print("  'clear'         - Clear conversation history")
    print("  'help'          - Show this message")
    print()


# =============================================================================
# INPUT HANDLING
# =============================================================================

def get_pdf_path() -> str:
    """
    Prompt user for a PDF file path and validate it.

    This function:
    1. Asks for a file path
    2. Cleans up the input (removes quotes, whitespace)
    3. Validates the file exists and is a PDF
    4. Loops until valid input or user quits

    Returns:
        str: Valid path to a PDF file

    User Experience Notes:
        - Handles copy-pasted paths with quotes
        - Gives helpful error messages
        - Allows empty input to retry
    """
    while True:
        print("Enter the path to your PDF file")
        print("(or 'quit' to exit)")
        path = input("> ").strip()

        # Check for quit command
        if path.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            sys.exit(0)

        # Remove surrounding quotes (common when copy-pasting paths)
        # Windows Explorer often adds quotes when copying paths
        path = path.strip('"').strip("'")

        # Validate: not empty
        if not path:
            print("Please enter a file path.\n")
            continue

        # Validate: file exists
        if not os.path.exists(path):
            print(f"\nFile not found: {path}")
            print("Please check the path and try again.\n")
            continue

        # Validate: is a PDF
        if not path.lower().endswith('.pdf'):
            print("\nPlease provide a PDF file (.pdf extension).")
            print("Other formats are not supported in this version.\n")
            continue

        return path


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def process_document(pdf_path: str):
    """
    Load and process a PDF document through the RAG pipeline.

    This is where the "magic" happens before you can ask questions:

    Steps:
        1. Load PDF and split into chunks
        2. Create embeddings for each chunk
        3. Store in vector database
        4. Create the QA chain

    Args:
        pdf_path: Path to the PDF file

    Returns:
        QAChain: Ready to answer questions

    Time Expectations:
        - First run: 30-60 seconds (downloads embedding model)
        - Subsequent: 5-15 seconds depending on document size
        - Larger documents = more chunks = longer processing

    What's Happening Behind the Scenes:
        1. PyPDF reads the PDF file
        2. Text is extracted from each page
        3. Text is split into ~1000 character chunks with overlap
        4. Each chunk is converted to a 384-dimension vector
        5. Vectors are stored in ChromaDB
        6. QAChain is initialized with the vector store
    """
    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    print("-" * 40)

    # ===== STEP 1: LOAD AND SPLIT =====
    print("\n[1/3] Loading PDF and splitting into chunks...")
    print("      (Extracting text, creating chunks with overlap)")

    chunks = load_and_split_pdf(pdf_path)

    print(f"      Done! Created {len(chunks)} text chunks")
    print(f"      Average chunk size: ~{sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    # ===== STEP 2: CREATE EMBEDDINGS =====
    print("\n[2/3] Creating embeddings and storing in vector database...")
    print("      (Converting text to vectors for semantic search)")
    print("      This may take 30+ seconds on first run...")

    # This step:
    # - Downloads the embedding model (first time only)
    # - Converts each chunk to a 384-dimensional vector
    # - Stores vectors + text + metadata in ChromaDB
    vector_store = create_vector_store(chunks)

    print("      Done! Embeddings created and stored")

    # ===== STEP 3: CREATE QA CHAIN =====
    print("\n[3/3] Initializing Q&A system...")
    print("      (Setting up Claude connection and retriever)")

    qa_chain = create_qa_chain(vector_store)

    print("      Done! Ready to answer questions")
    print("-" * 40)

    return qa_chain


# =============================================================================
# MAIN CHAT LOOP
# =============================================================================

def chat_loop(qa_chain):
    """
    Main interaction loop for asking questions.

    This is where users spend most of their time:
    - Type questions
    - Get answers
    - Ask follow-ups

    Args:
        qa_chain: The initialized QAChain from process_document()

    Returns:
        str: 'new' if user wants to load a new document
        None: if user wants to quit

    Features:
        - Conversation memory (follow-up questions work)
        - Error handling (graceful API error recovery)
        - Keyboard interrupt handling (Ctrl+C)
    """
    print("\n" + "=" * 60)
    print("Document loaded! You can now ask questions.")
    print_help()
    print("=" * 60)

    while True:
        try:
            # Get user input
            print()  # Blank line for readability
            question = input("You: ").strip()

            # Skip empty input
            if not question:
                continue

            # ===== HANDLE COMMANDS =====

            # Quit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                return None

            # Load new document
            if question.lower() == 'new':
                print("\nLoading new document...")
                return 'new'

            # Clear conversation history
            if question.lower() == 'clear':
                qa_chain.clear_history()
                print("Conversation history cleared. Ask a new question!")
                continue

            # Show help
            if question.lower() == 'help':
                print_help()
                continue

            # ===== ASK THE QUESTION =====

            print("\nSearching document and generating answer...")

            # This is where the RAG magic happens:
            # 1. Question -> vector
            # 2. Find similar chunks in vector store
            # 3. Send chunks + question to Claude
            # 4. Get answer
            result = ask_question(qa_chain, question)

            # Format and display the response
            response = format_response(result)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
            continue

        except Exception as e:
            # Handle other errors (API issues, etc.)
            print(f"\nError: {str(e)}")
            print("Please try again. If the problem persists, check your API key.")
            continue


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the application.

    This function orchestrates the entire user experience:
    1. Show welcome screen
    2. Get PDF path from user
    3. Process the document
    4. Enter chat loop
    5. Handle 'new document' requests
    6. Exit gracefully

    The outer while loop allows users to load multiple
    documents in one session without restarting.
    """
    # Clear screen and show welcome message
    clear_screen()
    print_header()

    # Main application loop
    # Allows loading multiple documents without restarting
    while True:
        try:
            # Get PDF path from user
            pdf_path = get_pdf_path()

            # Process the document (load, embed, store)
            qa_chain = process_document(pdf_path)

            # Enter the chat loop
            result = chat_loop(qa_chain)

            # Check if user wants to load a new document
            if result == 'new':
                clear_screen()
                print_header()
                print("Ready to load a new document.\n")
                continue
            else:
                # User wants to quit
                break

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again or check your setup.\n")
            continue


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    This block only runs when the script is executed directly,
    not when imported as a module.

    Example:
        python main.py          -> runs main()
        from main import main   -> does NOT run main()
    """
    main()


# =============================================================================
# EDUCATIONAL NOTES FOR DEVELOPERS
# =============================================================================
"""
EXTENDING THE CLI
-----------------

To add new commands, add handlers in the chat_loop function:

    if question.lower() == 'summary':
        # Generate a document summary
        result = ask_question(qa_chain, "Summarize this document in 3 bullet points")
        print(format_response(result))
        continue

    if question.lower() == 'sources':
        # Show all available source chunks
        for doc in qa_chain.vector_store.similarity_search("", k=10):
            print(f"Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
        continue

ADDING A WEB INTERFACE
----------------------

To add a web UI, create a new file (e.g., app.py) using Streamlit:

    import streamlit as st
    from src.document_loader import load_and_split_pdf
    from src.vector_store import create_vector_store
    from src.qa_chain import create_qa_chain, ask_question

    st.title("Document Q&A")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Process
        chunks = load_and_split_pdf("temp.pdf")
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

        # Chat interface
        question = st.text_input("Ask a question")
        if question:
            result = ask_question(qa_chain, question)
            st.write(result['answer'])

    Run with: streamlit run app.py

ERROR HANDLING BEST PRACTICES
-----------------------------

1. API Errors:
   - Retry with exponential backoff (already implemented in qa_chain)
   - Show user-friendly messages, not stack traces

2. File Errors:
   - Validate paths before processing
   - Handle permission errors gracefully

3. Memory Errors:
   - For very large documents, process in batches
   - Clear old vector stores when loading new documents

4. Network Errors:
   - Cache the embedding model locally
   - Consider offline fallbacks for embeddings

PERFORMANCE TIPS
----------------

1. First-Run Optimization:
   - Embedding model download is cached (~100MB)
   - Subsequent runs are much faster

2. Large Documents:
   - Consider showing a progress bar for embedding
   - Process pages in batches for very large PDFs

3. Memory Usage:
   - ChromaDB is efficient but grows with documents
   - Clear chroma_db/ folder to reset

4. Response Time:
   - API latency is the main bottleneck
   - Consider caching common questions
"""
