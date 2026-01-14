"""
Document Loader Module
======================

This module handles the first two steps of the RAG (Retrieval Augmented Generation) pipeline:
1. Loading documents (extracting text from PDFs)
2. Splitting text into smaller chunks

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

from langchain_community.document_loaders import PyPDFLoader
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
