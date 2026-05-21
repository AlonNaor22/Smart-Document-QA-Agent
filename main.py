#!/usr/bin/env python3
"""CLI entry point for the Smart Document Q&A Agent."""

import os
import sys

from src.document_loader import (
    load_and_split_document,
    load_and_split_folder,
    is_folder,
    SUPPORTED_EXTENSIONS
)
from src.vector_store import create_vector_store, CHROMA_PERSIST_DIR
from src.qa_chain import create_qa_chain, ask_question, format_response


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Display the application header/welcome message."""
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
    """Display available commands to the user."""
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


def get_document_path() -> str:
    """Prompt the user for a document path and return a validated file or folder path."""
    formats_display = ', '.join(SUPPORTED_EXTENSIONS)

    while True:
        print(f"Enter the path to your document OR folder")
        print(f"  - Single file: path to a {formats_display} file")
        print(f"  - Multiple files: path to a folder containing documents")
        print("(or 'quit' to exit)")
        path = input("> ").strip()

        if path.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            sys.exit(0)

        path = path.strip('"').strip("'")

        if not path:
            print("Please enter a path.\n")
            continue

        if not os.path.exists(path):
            print(f"\nPath not found: {path}")
            print("Please check the path and try again.\n")
            continue

        if os.path.isdir(path):
            return path

        file_extension = os.path.splitext(path)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            print(f"\nUnsupported file format: {file_extension}")
            print(f"Supported formats: {formats_display}\n")
            continue

        return path


def process_document(path: str):
    """Load and process a file or folder through the RAG pipeline, returning a ready QA chain."""
    is_folder_path = is_folder(path)

    if is_folder_path:
        print(f"\nProcessing folder: {os.path.basename(path)}")
    else:
        print(f"\nProcessing: {os.path.basename(path)}")
    print("-" * 50)

    if is_folder_path:
        print("\n[1/3] Loading ALL documents from folder...")
        print("      (Scanning for PDF, DOCX, TXT, MD files)")

        chunks, summary = load_and_split_folder(path)

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

    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"      Average chunk size: ~{avg_size} characters")

    print("\n[2/3] Creating embeddings and storing in vector database...")
    print("      (Clearing old data and creating fresh embeddings)")
    print("      This may take 30+ seconds on first run...")

    vector_store = create_vector_store(chunks)

    print("      Done! Embeddings created and stored")

    print("\n[3/3] Initializing Q&A system...")
    print("      (Setting up Claude connection and retriever)")

    qa_chain = create_qa_chain(vector_store)

    print("      Done! Ready to answer questions")
    print("-" * 50)

    if is_folder_path:
        print(f"\nYou can now ask questions about ALL {summary['successful']} documents!")
        print("The AI will search across all files to find relevant information.")

    return qa_chain


def chat_loop(qa_chain):
    """Run the interactive Q&A loop; returns 'new' to load a new document, None to quit."""
    print("\n" + "=" * 60)
    print("Ready! You can now ask questions about your document(s).")
    print_help()
    print("=" * 60)

    while True:
        try:
            print()
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                return None

            if question.lower() == 'new':
                print("\nLoading new document...")
                return 'new'

            if question.lower() == 'clear':
                qa_chain.clear_history()
                print("Conversation history cleared. Ask a new question!")
                continue

            if question.lower() == 'help':
                print_help()
                continue

            print("\nSearching document and generating answer...")

            result = ask_question(qa_chain, question)

            response = format_response(result)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
            continue

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again. If the problem persists, check your API key.")
            continue


def main():
    """Run the main application loop: show welcome, get document path, process, and chat."""
    clear_screen()
    print_header()

    # Allows loading multiple documents in one session without restarting
    while True:
        try:
            file_path = get_document_path()

            qa_chain = process_document(file_path)

            result = chat_loop(qa_chain)

            if result == 'new':
                clear_screen()
                print_header()
                print("Ready to load a new document.\n")
                continue
            else:
                break

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again or check your setup.\n")
            continue


if __name__ == "__main__":
    main()
