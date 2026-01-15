#!/usr/bin/env python3
"""
Smart Document Q&A Agent - Main Entry Point
============================================

This is the command-line interface (CLI) for the Document Q&A system.
Run this file to start an interactive session where you can:
1. Load a document (PDF, DOCX, TXT, or MD)
2. Ask questions about its content
3. Get AI-powered answers with source references

SUPPORTED FILE FORMATS
----------------------
- PDF  (.pdf)  - Adobe PDF documents
- DOCX (.docx) - Microsoft Word documents
- TXT  (.txt)  - Plain text files
- MD   (.md)   - Markdown files

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
2. You enter a path to a document file
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
from src.document_loader import (
    load_and_split_document,
    load_and_split_folder,
    is_folder,
    SUPPORTED_EXTENSIONS
)
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
    print("       Ask questions about your documents")
    print("=" * 60)
    print()
    print("Supports: PDF, DOCX, TXT, MD")
    print("You can load a SINGLE FILE or an entire FOLDER!")
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
    print("  [any question]  - Ask about your document(s)")
    print("  'quit' or 'exit'- Leave the application")
    print("  'new'           - Load different document(s)")
    print("  'clear'         - Clear conversation history")
    print("  'help'          - Show this message")
    print()
    print("Tip: If you loaded multiple documents, your questions")
    print("     will search across ALL of them!")
    print()


# =============================================================================
# INPUT HANDLING
# =============================================================================

def get_document_path() -> str:
    """
    Prompt user for a document file OR folder path and validate it.

    This function:
    1. Asks for a file or folder path
    2. Cleans up the input (removes quotes, whitespace)
    3. Validates the path exists
    4. For files: checks it's a supported format
    5. For folders: accepts it (will scan for documents inside)
    6. Loops until valid input or user quits

    Supported formats: PDF, DOCX, TXT, MD

    Returns:
        str: Valid path to a document file OR a folder

    User Experience Notes:
        - Handles copy-pasted paths with quotes
        - Gives helpful error messages
        - Allows empty input to retry
    """
    # Create a nice display of supported formats
    formats_display = ', '.join(SUPPORTED_EXTENSIONS)

    while True:
        print(f"Enter the path to your document OR folder")
        print(f"  - Single file: path to a {formats_display} file")
        print(f"  - Multiple files: path to a folder containing documents")
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
            print("Please enter a path.\n")
            continue

        # Validate: path exists
        if not os.path.exists(path):
            print(f"\nPath not found: {path}")
            print("Please check the path and try again.\n")
            continue

        # If it's a folder, accept it (we'll scan for documents inside)
        if os.path.isdir(path):
            return path

        # If it's a file, validate the format
        file_extension = os.path.splitext(path)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            print(f"\nUnsupported file format: {file_extension}")
            print(f"Supported formats: {formats_display}\n")
            continue

        return path


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def process_document(path: str):
    """
    Load and process document(s) through the RAG pipeline.

    This function handles BOTH single files AND folders:
    - Single file: Loads just that document
    - Folder: Loads ALL supported documents inside

    This is where the "magic" happens before you can ask questions:

    Steps:
        1. Load document(s) and split into chunks
        2. Create embeddings for each chunk
        3. Store in vector database
        4. Create the QA chain

    Args:
        path: Path to a document file OR a folder containing documents

    Returns:
        QAChain: Ready to answer questions

    Time Expectations:
        - First run: 30-60 seconds (downloads embedding model)
        - Subsequent: 5-15 seconds depending on document size
        - More documents = more chunks = longer processing
    """
    # Check if path is a folder or single file
    is_folder_path = is_folder(path)

    if is_folder_path:
        print(f"\nProcessing folder: {os.path.basename(path)}")
    else:
        print(f"\nProcessing: {os.path.basename(path)}")
    print("-" * 50)

    # ===== STEP 1: LOAD AND SPLIT =====
    if is_folder_path:
        print("\n[1/3] Loading ALL documents from folder...")
        print("      (Scanning for PDF, DOCX, TXT, MD files)")

        chunks, summary = load_and_split_folder(path)

        # Show what was loaded
        print(f"\n      Files found: {summary['total_files']}")
        print(f"      Successfully loaded: {summary['successful']}")

        if summary['files_loaded']:
            print("\n      Loaded files:")
            for file_info in summary['files_loaded']:
                print(f"        - {file_info['filename']}")

        if summary['files_failed']:
            print(f"\n      Failed to load: {summary['failed']}")
            for file_info in summary['files_failed']:
                print(f"        - {file_info['filename']}: {file_info['error']}")

        print(f"\n      Total chunks created: {len(chunks)}")

    else:
        print("\n[1/3] Loading document and splitting into chunks...")
        print("      (Extracting text, creating chunks with overlap)")

        chunks = load_and_split_document(path)

        print(f"      Done! Created {len(chunks)} text chunks")

    # Show average chunk size
    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"      Average chunk size: ~{avg_size} characters")

    # ===== STEP 2: CREATE EMBEDDINGS =====
    print("\n[2/3] Creating embeddings and storing in vector database...")
    print("      (Clearing old data and creating fresh embeddings)")
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
    print("-" * 50)

    # Show helpful message for folder loading
    if is_folder_path:
        print(f"\nYou can now ask questions about ALL {summary['successful']} documents!")
        print("The AI will search across all files to find relevant information.")

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
    print("Ready! You can now ask questions about your document(s).")
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
            # Get document path from user (supports PDF, DOCX, TXT, MD)
            file_path = get_document_path()

            # Process the document (load, embed, store)
            qa_chain = process_document(file_path)

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
