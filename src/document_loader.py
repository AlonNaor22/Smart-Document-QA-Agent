"""
Document Loader Module
======================

This module handles the first two steps of the RAG (Retrieval Augmented Generation) pipeline:
1. Loading documents (extracting text from PDFs, DOCX, TXT, MD files)
2. Splitting text into smaller chunks

SUPPORTED FILE FORMATS
----------------------
- PDF  (.pdf)  - Uses PyPDFLoader, extracts text page by page
- DOCX (.docx) - Uses Docx2txtLoader for Microsoft Word documents
- TXT  (.txt)  - Uses TextLoader for plain text files
- MD   (.md)   - Uses TextLoader for Markdown files (treated as plain text)

WHY DO WE NEED THIS?
--------------------
AI models like Claude have a "context window" - a limit on how much text they can
process at once. For example:
- Claude Sonnet: ~200,000 tokens (roughly 150,000 words)

While this seems large, we DON'T want to send entire documents because:
1. Cost: You pay per token. Sending unnecessary text wastes money.
2. Accuracy: Finding a needle in a haystack is hard. Smaller, relevant chunks = better answers.
3. Speed: Less text = faster processing.

THE CHUNKING STRATEGY
---------------------
We use "RecursiveCharacterTextSplitter" which is smart about where it splits:

1. First, tries to split at paragraph breaks (\\n\\n)
2. If chunks are still too big, splits at line breaks (\\n)
3. Then at spaces (between words)
4. Last resort: splits mid-word (rarely needed)

This preserves meaning better than just cutting every N characters.

CHUNK OVERLAP EXPLAINED
-----------------------
Imagine this text split into 2 chunks without overlap:
    Chunk 1: "The meeting is scheduled for"
    Chunk 2: "Tuesday at 3pm in Room 101"

If someone asks "When is the meeting?", Chunk 2 has the answer but lacks context.

With overlap:
    Chunk 1: "The meeting is scheduled for Tuesday"
    Chunk 2: "scheduled for Tuesday at 3pm in Room 101"

Now Chunk 2 has enough context to answer the question properly!

USAGE EXAMPLE
-------------
    from src.document_loader import load_and_split_pdf

    # Load and process a PDF
    chunks = load_and_split_pdf("path/to/document.pdf")

    # Each chunk is a Document object with:
    # - page_content: The actual text
    # - metadata: Info like page number, source file, etc.

    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0].page_content[:100]}...")
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import our centralized configuration
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> list:
    """
    Load a PDF file and extract text from each page.

    This function uses PyPDFLoader which:
    - Opens the PDF file
    - Extracts text from each page
    - Returns a list of Document objects (one per page)
    - Preserves metadata (page numbers, source file)

    Args:
        file_path: Path to the PDF file (absolute or relative)

    Returns:
        List of Document objects, each containing:
        - page_content: Text extracted from one page
        - metadata: {"source": file_path, "page": page_number}

    Example:
        >>> docs = load_pdf("report.pdf")
        >>> print(len(docs))  # Number of pages
        10
        >>> print(docs[0].metadata)
        {"source": "report.pdf", "page": 0}

    Note:
        - Page numbers are 0-indexed (first page is page 0)
        - Some PDFs with images/scans may extract poorly
        - For scanned PDFs, you'd need OCR (not covered here)
    """
    # PyPDFLoader is a LangChain wrapper around the pypdf library
    loader = PyPDFLoader(file_path)

    # .load() reads the entire PDF and returns all pages
    # For very large PDFs, consider .lazy_load() which yields one page at a time
    pages = loader.load()

    return pages


def load_docx(file_path: str) -> list:
    """
    Load a Word document (.docx) and extract its text.

    Uses Docx2txtLoader which extracts all text content from the document.
    Unlike PDFs, Word documents don't have "pages" in the same way,
    so the entire document is returned as a single Document object.

    Args:
        file_path: Path to the .docx file

    Returns:
        List containing one Document object with all text

    Note:
        - Formatting (bold, italic, etc.) is lost - only text is extracted
        - Tables are converted to plain text
        - Images are ignored
    """
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    return documents


def load_text(file_path: str) -> list:
    """
    Load a plain text file (.txt or .md).

    This is the simplest loader - it just reads the file content.
    Works for any plain text format including Markdown.

    Args:
        file_path: Path to the text file

    Returns:
        List containing one Document object with file content

    Note:
        - Encoding is auto-detected (usually UTF-8)
        - The entire file is loaded as one Document
        - Metadata includes the source file path
    """
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    return documents


# List of supported file extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md']


def get_file_extension(file_path: str) -> str:
    """
    Get the lowercase file extension from a path.

    Example:
        >>> get_file_extension("Report.PDF")
        '.pdf'
        >>> get_file_extension("notes.TXT")
        '.txt'
    """
    return os.path.splitext(file_path)[1].lower()


def load_document(file_path: str) -> list:
    """
    Load a document of any supported type.

    This is the main entry point for loading documents. It automatically
    detects the file type based on extension and uses the appropriate loader.

    Supported formats:
        - .pdf  - PDF documents
        - .docx - Microsoft Word documents
        - .txt  - Plain text files
        - .md   - Markdown files

    Args:
        file_path: Path to the document file

    Returns:
        List of Document objects with extracted text

    Raises:
        ValueError: If the file extension is not supported

    Example:
        >>> docs = load_document("report.pdf")      # Works!
        >>> docs = load_document("notes.docx")      # Works!
        >>> docs = load_document("readme.md")       # Works!
        >>> docs = load_document("data.xlsx")       # Error - not supported
    """
    extension = get_file_extension(file_path)

    if extension == '.pdf':
        return load_pdf(file_path)
    elif extension == '.docx':
        return load_docx(file_path)
    elif extension in ['.txt', '.md']:
        return load_text(file_path)
    else:
        supported = ', '.join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type: {extension}\n"
            f"Supported formats: {supported}"
        )


def split_into_chunks(
    documents: list,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Split documents into smaller, overlapping chunks.

    This is crucial for effective retrieval. The chunking strategy affects:
    - Search accuracy: Well-sized chunks = better matches
    - Answer quality: Too small = fragmented context, too large = noise
    - Cost: Smaller chunks = more of them, but only relevant ones are used

    Args:
        documents: List of Document objects from load_pdf()
        chunk_size: Maximum characters per chunk (default from config)
        chunk_overlap: Characters to repeat between chunks (default from config)

    Returns:
        List of Document objects (more than input, since pages are split)
        Each chunk preserves the original metadata plus chunk info.

    Example:
        >>> docs = load_pdf("report.pdf")  # 10 pages
        >>> chunks = split_into_chunks(docs)
        >>> print(len(chunks))  # Probably 30-50 chunks
        42

    The Math:
        If a page has 3000 characters, chunk_size=1000, overlap=200:
        - Chunk 1: chars 0-1000
        - Chunk 2: chars 800-1800 (starts 200 before end of chunk 1)
        - Chunk 3: chars 1600-2600
        - Chunk 4: chars 2400-3000
        Result: 4 chunks from one page
    """
    # RecursiveCharacterTextSplitter is the recommended splitter for most use cases
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        # len counts characters. Could also use a token counter for more accuracy.
        length_function=len,

        # Separators are tried in order. First one that creates small enough chunks wins.
        # This preserves document structure (paragraphs > lines > words > characters)
        separators=[
            "\n\n",  # Paragraph breaks (best split point)
            "\n",    # Line breaks
            " ",     # Spaces (word boundaries)
            ""       # Character-level (last resort)
        ]
    )

    # split_documents() handles the Document objects, preserving metadata
    chunks = text_splitter.split_documents(documents)

    return chunks


def load_and_split_pdf(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Convenience function: Load a PDF and split it in one step.

    This is the main function you'll use in most cases. It combines
    loading and splitting into a single call.

    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        List of Document chunks ready for embedding

    Example:
        >>> chunks = load_and_split_pdf("contract.pdf")
        >>> print(f"Ready to process {len(chunks)} chunks")
        Ready to process 25 chunks

        >>> # Each chunk has content and metadata
        >>> chunk = chunks[0]
        >>> print(chunk.page_content[:50])
        "This agreement is entered into on January 1st..."
        >>> print(chunk.metadata)
        {"source": "contract.pdf", "page": 0}

    Pipeline Position:
        This is Step 1 of the RAG pipeline:
        [load_and_split_pdf] -> embeddings -> vector store -> retrieval -> LLM
    """
    # Step 1: Extract text from PDF pages
    documents = load_pdf(file_path)

    # Step 2: Split into smaller chunks
    chunks = split_into_chunks(documents, chunk_size, chunk_overlap)

    return chunks


def load_and_split_document(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Load ANY supported document type and split it into chunks.

    This is the NEW main function that replaces load_and_split_pdf.
    It automatically detects the file type and processes it appropriately.

    Args:
        file_path: Path to any supported document (PDF, DOCX, TXT, MD)
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        List of Document chunks ready for embedding

    Example:
        >>> # All of these work the same way:
        >>> chunks = load_and_split_document("report.pdf")
        >>> chunks = load_and_split_document("notes.docx")
        >>> chunks = load_and_split_document("readme.md")

    Pipeline Position:
        [load_and_split_document] -> embeddings -> vector store -> retrieval -> LLM
    """
    # Step 1: Load document (auto-detects file type)
    documents = load_document(file_path)

    # Step 2: Split into smaller chunks
    chunks = split_into_chunks(documents, chunk_size, chunk_overlap)

    return chunks


# =============================================================================
# MULTIPLE DOCUMENT SUPPORT
# =============================================================================

def scan_folder_for_documents(folder_path: str) -> list:
    """
    Scan a folder and return paths to all supported document files.

    This function looks through a folder (not recursively) and finds
    all files with supported extensions (.pdf, .docx, .txt, .md).

    Args:
        folder_path: Path to the folder to scan

    Returns:
        List of full file paths to supported documents

    Example:
        >>> files = scan_folder_for_documents("./data")
        >>> print(files)
        ['./data/report.pdf', './data/notes.txt', './data/readme.md']

    Note:
        - Only scans the top level (not subfolders)
        - Skips files with unsupported extensions
        - Returns empty list if folder is empty or has no supported files
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a valid folder: {folder_path}")

    supported_files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if file has a supported extension
        extension = get_file_extension(file_path)
        if extension in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)

    return sorted(supported_files)  # Sort for consistent ordering


def load_multiple_documents(file_paths: list) -> tuple:
    """
    Load multiple documents and combine their content.

    This function loads several documents at once and merges them
    into a single list, ready for chunking and embedding.

    Args:
        file_paths: List of paths to document files

    Returns:
        Tuple of (combined_documents, load_summary)
        - combined_documents: List of all Document objects
        - load_summary: Dict with stats about what was loaded

    Example:
        >>> paths = ['report.pdf', 'notes.txt']
        >>> docs, summary = load_multiple_documents(paths)
        >>> print(summary)
        {'total_files': 2, 'successful': 2, 'failed': 0, 'files_loaded': [...]}
    """
    all_documents = []
    files_loaded = []
    files_failed = []

    for file_path in file_paths:
        try:
            docs = load_document(file_path)
            all_documents.extend(docs)
            files_loaded.append({
                'path': file_path,
                'filename': os.path.basename(file_path),
                'pages': len(docs)
            })
        except Exception as e:
            files_failed.append({
                'path': file_path,
                'filename': os.path.basename(file_path),
                'error': str(e)
            })

    summary = {
        'total_files': len(file_paths),
        'successful': len(files_loaded),
        'failed': len(files_failed),
        'files_loaded': files_loaded,
        'files_failed': files_failed
    }

    return all_documents, summary


def load_and_split_folder(
    folder_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> tuple:
    """
    Load ALL supported documents from a folder and split into chunks.

    This is the main function for multi-document support. It:
    1. Scans the folder for supported files
    2. Loads all documents
    3. Splits everything into chunks
    4. Returns chunks ready for embedding

    Args:
        folder_path: Path to folder containing documents
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        Tuple of (chunks, summary)
        - chunks: List of Document chunks from ALL files
        - summary: Dict with loading statistics

    Example:
        >>> chunks, summary = load_and_split_folder("./data")
        >>> print(f"Loaded {summary['successful']} files")
        >>> print(f"Created {len(chunks)} total chunks")

    Pipeline Position:
        [load_and_split_folder] -> embeddings -> vector store -> retrieval -> LLM
    """
    # Step 1: Find all supported files
    file_paths = scan_folder_for_documents(folder_path)

    if not file_paths:
        raise ValueError(
            f"No supported documents found in: {folder_path}\n"
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Step 2: Load all documents
    documents, summary = load_multiple_documents(file_paths)

    if not documents:
        raise ValueError("Failed to load any documents from the folder.")

    # Step 3: Split into chunks
    chunks = split_into_chunks(documents, chunk_size, chunk_overlap)

    # Add chunk count to summary
    summary['total_chunks'] = len(chunks)

    return chunks, summary


def is_folder(path: str) -> bool:
    """
    Check if a path is a folder (directory).

    Args:
        path: Path to check

    Returns:
        True if path is a folder, False if it's a file
    """
    return os.path.isdir(path)


# =============================================================================
# EDUCATIONAL NOTES FOR DEVELOPERS
# =============================================================================
"""
EXTENDING THIS MODULE
---------------------

To add support for other file types, create similar functions:

def load_docx(file_path: str) -> list:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    return loader.load()

def load_txt(file_path: str) -> list:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path)
    return loader.load()

ALTERNATIVE SPLITTERS
---------------------

LangChain offers other splitters for specific use cases:

1. TokenTextSplitter: Splits by tokens instead of characters
   - More accurate for LLM context limits
   - Requires tokenizer (adds complexity)

2. MarkdownTextSplitter: Respects Markdown structure
   - Great for documentation
   - Keeps headers with their content

3. CodeTextSplitter: Splits code intelligently
   - Respects function/class boundaries
   - Supports multiple programming languages

DEBUGGING TIPS
--------------

If retrieval quality is poor, try:
1. Print chunks to see what's being created:
   for i, chunk in enumerate(chunks[:3]):
       print(f"Chunk {i}: {chunk.page_content[:100]}...")

2. Adjust chunk_size:
   - Increase if answers seem fragmented
   - Decrease if irrelevant content is included

3. Check metadata is preserved:
   print(chunks[0].metadata)  # Should show source and page
"""
