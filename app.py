"""
Smart Document Q&A Agent - Web Interface
=========================================

This is the Streamlit web interface for the Document Q&A system.
It provides a visual, user-friendly way to:
1. Upload documents (PDF, DOCX, TXT, MD)
2. Ask questions in a chat interface
3. See answers with sources and confidence scores

HOW TO RUN
----------
From the project directory:
    streamlit run app.py

This will open a browser window with the application.

STREAMLIT BASICS
----------------
- Streamlit reruns the entire script on each interaction
- We use st.session_state to persist data between reruns
- Components like st.file_uploader, st.chat_input create the UI
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Import our existing modules - we reuse all the logic we built!
from src.document_loader import (
    load_and_split_document,
    SUPPORTED_EXTENSIONS,
    get_file_extension
)
from src.vector_store import create_vector_store
from src.qa_chain import create_qa_chain

# Load environment variables (for ANTHROPIC_API_KEY)
load_dotenv()


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# Must be the first Streamlit command
st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """
    Initialize session state variables.

    Session state persists data between Streamlit reruns.
    Without this, all variables would reset on each interaction!
    """
    # QA chain (the brain of our system)
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Chat history for display
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Track loaded files
    if 'loaded_files' not in st.session_state:
        st.session_state.loaded_files = []

    # Processing state
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def process_uploaded_files(uploaded_files) -> bool:
    """
    Process uploaded files and create the QA chain.

    Args:
        uploaded_files: List of UploadedFile objects from Streamlit

    Returns:
        True if successful, False otherwise
    """
    if not uploaded_files:
        return False

    all_chunks = []
    loaded_files = []

    # Create a temporary directory to save uploaded files
    # (LangChain loaders need file paths, not file objects)
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Check file extension
            ext = get_file_extension(uploaded_file.name)
            if ext not in SUPPORTED_EXTENSIONS:
                st.warning(f"Skipped unsupported file: {uploaded_file.name}")
                continue

            # Save to temp file
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Load and split the document
                chunks = load_and_split_document(temp_path)

                # Update source metadata to use original filename
                for chunk in chunks:
                    chunk.metadata['source'] = uploaded_file.name

                all_chunks.extend(chunks)
                loaded_files.append({
                    'name': uploaded_file.name,
                    'chunks': len(chunks)
                })

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if not all_chunks:
        st.error("No documents were successfully processed.")
        return False

    # Create vector store and QA chain
    try:
        with st.spinner("Creating embeddings... (this may take a moment)"):
            vector_store = create_vector_store(all_chunks)
            st.session_state.qa_chain = create_qa_chain(vector_store)
            st.session_state.loaded_files = loaded_files

        return True

    except Exception as e:
        st.error(f"Error creating QA system: {str(e)}")
        return False


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with file upload and info."""

    with st.sidebar:
        st.title("📄 Document Q&A")
        st.markdown("---")

        # File uploader
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD"
        )

        # Process button
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                st.session_state.is_processing = True

                with st.spinner("Processing documents..."):
                    success = process_uploaded_files(uploaded_files)

                if success:
                    st.success(f"✅ Loaded {len(st.session_state.loaded_files)} document(s)!")
                    # Clear chat when new documents are loaded
                    st.session_state.messages = []
                    st.rerun()

                st.session_state.is_processing = False

        st.markdown("---")

        # Show loaded files
        if st.session_state.loaded_files:
            st.subheader("📚 Loaded Documents")
            for file_info in st.session_state.loaded_files:
                st.markdown(f"• **{file_info['name']}** ({file_info['chunks']} chunks)")

            # Total chunks
            total_chunks = sum(f['chunks'] for f in st.session_state.loaded_files)
            st.caption(f"Total: {total_chunks} searchable chunks")

            st.markdown("---")

            # Clear button
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.qa_chain = None
                st.session_state.messages = []
                st.session_state.loaded_files = []
                st.rerun()

        # Help section
        st.markdown("---")
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Upload** one or more documents
        2. Click **Process Documents**
        3. **Ask questions** in the chat
        4. Get answers with **sources**!
        """)

        st.markdown("---")
        st.caption("Built with LangChain + Claude + Streamlit")


def render_chat():
    """Render the main chat interface."""

    # Header
    st.title("💬 Chat with Your Documents")

    # Check if documents are loaded
    if not st.session_state.qa_chain:
        st.info("👈 Upload documents in the sidebar to get started!")

        # Show example
        with st.expander("📖 Example Questions"):
            st.markdown("""
            Once you upload documents, you can ask questions like:
            - "What is this document about?"
            - "Summarize the main points"
            - "What are the key findings?"
            - "Tell me more about [specific topic]"
            """)
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"], message.get("scores", []))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.ask(prompt)
                    answer = result.get("answer", "I couldn't generate an answer.")
                    sources = result.get("source_documents", [])
                    scores = result.get("confidence_scores", [])

                    st.markdown(answer)
                    render_sources(sources, scores)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "scores": scores
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def render_sources(sources, scores):
    """
    Render source documents with confidence scores.

    Args:
        sources: List of Document objects
        scores: List of confidence scores (percentages)
    """
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} chunks found)", expanded=False):
        for i, doc in enumerate(sources):
            # Get metadata
            filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', None)

            # Get confidence score
            score = scores[i] if i < len(scores) else None

            # Build header
            header_parts = [f"**{filename}**"]
            if page is not None:
                header_parts.append(f"Page {page}")
            if score is not None:
                # Color code based on score
                if score >= 80:
                    score_color = "green"
                elif score >= 60:
                    score_color = "orange"
                else:
                    score_color = "red"
                header_parts.append(f":{score_color}[{score}% match]")

            header = " | ".join(header_parts)
            st.markdown(header)

            # Show preview
            preview = doc.page_content[:300].replace("\n", " ")
            if len(doc.page_content) > 300:
                preview += "..."
            st.caption(f'"{preview}"')

            if i < len(sources) - 1:
                st.markdown("---")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""

    # Initialize session state
    init_session_state()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("⚠️ ANTHROPIC_API_KEY not found!")
        st.markdown("""
        Please create a `.env` file in the project root with:
        ```
        ANTHROPIC_API_KEY=your_api_key_here
        ```
        Get your API key from [console.anthropic.com](https://console.anthropic.com/)
        """)
        return

    # Render UI
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
