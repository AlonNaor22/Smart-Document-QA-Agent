# Smart Document Q&A Agent

An AI-powered document question-answering system using RAG (Retrieval Augmented Generation). Upload documents and ask questions about their content - get accurate answers with source references and confidence scores.

## Features

- **Multiple File Formats** - Support for PDF, DOCX, TXT, and Markdown files
- **Multi-Document Search** - Load entire folders and search across all documents at once
- **Confidence Scores** - See how relevant each source is (0-100%)
- **Source Highlighting** - Know exactly which file and section the answer came from
- **Conversation Memory** - Follow-up questions work naturally
- **Two Interfaces** - Command-line (CLI) or Web UI (Streamlit)

## Screenshots

### Web Interface
```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN CHAT AREA               │
│  ─────────                  │  ─────────────                 │
│  📄 Document Q&A            │  💬 Chat with Your Documents   │
│                             │                                │
│  [Upload Documents]         │  User: What is this about?     │
│   Drag & drop files         │                                │
│                             │  Assistant: Based on the       │
│  📚 Loaded Documents        │  document, it discusses...     │
│   • report.pdf (12 chunks)  │                                │
│   • notes.txt (8 chunks)    │  📚 Sources (4 chunks)         │
│                             │    report.pdf | 92% match      │
└─────────────────────────────────────────────────────────────┘
```

## How It Works (RAG Pipeline)

```
Documents (PDF, DOCX, TXT, MD)
     |
     v
[1. Text Extraction] --> Extract text from each file
     |
     v
[2. Chunking] --> Split into ~1000 char chunks with overlap
     |
     v
[3. Embeddings] --> Convert chunks to vectors (384 dimensions)
     |
     v
[4. Vector Store] --> Store in ChromaDB for fast similarity search
     |
     v
[5. User Question] --> Convert question to vector
     |
     v
[6. Retrieval] --> Find most similar chunks + confidence scores
     |
     v
[7. Generation] --> Send chunks + question to Claude
     |
     v
Answer with Sources & Confidence Scores
```

**Why RAG?**
- You can't feed a 500-page document directly to an AI (too large, too expensive)
- RAG finds only the relevant parts and sends those to the AI
- Results in accurate, grounded answers based on YOUR documents

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AlonNaor22/Smart-Document-QA-Agent.git
cd Smart-Document-QA-Agent
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your API Key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from [console.anthropic.com](https://console.anthropic.com/)

### 5. Run the Application

**Option A: Command Line Interface**
```bash
python main.py
```

**Option B: Web Interface (Recommended)**
```bash
streamlit run app.py
```

## Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Adobe PDF documents |
| Word | `.docx` | Microsoft Word documents |
| Text | `.txt` | Plain text files |
| Markdown | `.md` | Markdown files |

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `[any question]` | Ask about your document(s) |
| `quit` or `exit` | Leave the application |
| `new` | Load different document(s) |
| `clear` | Clear conversation history |
| `help` | Show available commands |

### Loading Documents

**Single File:**
```
Enter the path to your document OR folder
> C:\path\to\document.pdf
```

**Multiple Files (Folder):**
```
Enter the path to your document OR folder
> C:\path\to\folder
```

The system will automatically find and load all supported files in the folder.

## Docker

The image pre-bakes the `all-MiniLM-L6-v2` embedding model at build time, so the first query has no download delay.

```bash
docker compose up --build
```

Then open http://localhost:8501 in your browser.

- Documents placed in the `data/` folder are available inside the container
- The vector database persists in `chroma_db/` between restarts
- Make sure `.env` exists with `ANTHROPIC_API_KEY` before running

## Project Structure

```
Smart-Document-QA-Agent/
├── main.py                 # CLI entry point
├── app.py                  # Web UI (Streamlit)
├── requirements.txt        # Python dependencies
├── .env                    # API key (create this)
├── src/
│   ├── config.py           # All settings in one place
│   ├── document_loader.py  # Multi-format document loading
│   ├── vector_store.py     # Embeddings & ChromaDB
│   └── qa_chain.py         # Q&A logic with Claude
├── data/                   # Place your documents here
└── chroma_db/              # Vector database (auto-created)
```

## Configuration

All settings are in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 4 | Number of chunks to retrieve |
| `MODEL_NAME` | claude-sonnet-4-5 | Claude model to use |
| `TEMPERATURE` | 0.0 | Response randomness (0 = deterministic) |

## Technologies Used

- **[LangChain](https://python.langchain.com/)** - RAG pipeline orchestration
- **[Anthropic Claude](https://www.anthropic.com/)** - LLM for question answering
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[HuggingFace](https://huggingface.co/)** - Embeddings model (all-MiniLM-L6-v2)
- **[Streamlit](https://streamlit.io/)** - Web interface
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF text extraction
- **[docx2txt](https://github.com/ankushshah89/python-docx2txt)** - Word document extraction

## Understanding Confidence Scores

When you ask a question, each source chunk shows a confidence score:

| Score | Meaning | Color (Web UI) |
|-------|---------|----------------|
| 80-100% | Highly relevant - strong match | Green |
| 60-79% | Good relevance - likely useful | Orange |
| Below 60% | Lower relevance - may be tangential | Red |

## Learning Resources

This project is documented for learning purposes. Each source file contains:
- Detailed docstrings explaining the concepts
- Inline comments explaining WHY, not just WHAT
- Educational notes section with tips for extending

Key concepts covered:
- RAG (Retrieval Augmented Generation)
- Text embeddings and vector similarity
- Prompt engineering
- Conversation memory
- Multi-document retrieval

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

77 tests covering config validation, document chunking logic, folder scanning, and QA helper functions (context formatting, distance-to-confidence, text sanitization) — all run without hitting any external API.

## License

MIT License - Feel free to use this project for learning or as a starting point for your own applications.

## Author

Built as a portfolio project to demonstrate RAG implementation with LangChain and Claude.
