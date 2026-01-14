# Smart Document Q&A Agent

An AI-powered document question-answering system using RAG (Retrieval Augmented Generation). Upload a PDF and ask questions about its content - get accurate answers with source references.

## What It Does

- **Load PDF documents** - Extract and process text from any PDF
- **Ask questions in natural language** - "What are the main findings?" or "Summarize section 3"
- **Get AI-powered answers** - Claude analyzes relevant sections and responds
- **See source references** - Know exactly which parts of the document the answer came from
- **Have conversations** - Follow-up questions work thanks to conversation memory

## How It Works (RAG Pipeline)

```
PDF Document
     |
     v
[1. Text Extraction] --> Extract text from each page
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
[6. Retrieval] --> Find most similar chunks
     |
     v
[7. Generation] --> Send chunks + question to Claude
     |
     v
Answer with Sources
```

**Why RAG?**
- You can't feed a 500-page PDF directly to an AI (too large, too expensive)
- RAG finds only the relevant parts and sends those to the AI
- Results in accurate, grounded answers based on YOUR document

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

```bash
python main.py
```

## Commands

| Command | Description |
|---------|-------------|
| `[any question]` | Ask about the document |
| `quit` or `exit` | Leave the application |
| `new` | Load a different document |
| `clear` | Clear conversation history |
| `help` | Show available commands |

## Project Structure

```
Smart-Document-QA-Agent/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env                    # API key (create this)
├── src/
│   ├── config.py           # All settings in one place
│   ├── document_loader.py  # PDF loading & text chunking
│   ├── vector_store.py     # Embeddings & ChromaDB
│   └── qa_chain.py         # Q&A logic with Claude
├── data/                   # Place your PDFs here
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
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF text extraction

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

## License

MIT License - Feel free to use this project for learning or as a starting point for your own applications.

## Author

Built as a portfolio project to demonstrate RAG implementation with LangChain and Claude.
