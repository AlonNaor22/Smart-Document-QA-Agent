"""
QA Chain Module
===============

This is the "brain" of the application - it orchestrates the entire
question-answering process by connecting retrieval to the LLM (Claude).

THE RAG FLOW (Retrieval Augmented Generation)
---------------------------------------------

When a user asks a question, here's what happens:

1. RETRIEVE: Find relevant chunks from the vector store
   "What are the payment terms?" -> [chunk about payments, chunk about deadlines, ...]

2. AUGMENT: Add those chunks as context to the prompt
   "Here's the relevant info: [chunks]. Now answer: What are the payment terms?"

3. GENERATE: Send to Claude, get an answer
   Claude reads the context and generates: "The payment terms are net-30..."

WHY RAG? (vs. just asking the LLM)
----------------------------------

Without RAG:
- User: "What's in my contract?"
- LLM: "I don't have access to your contract" (or worse, makes something up)

With RAG:
- User: "What's in my contract?"
- System: [retrieves relevant chunks from YOUR contract]
- LLM: "Based on section 3.2 of your contract, the terms are..."

RAG grounds the LLM in YOUR data, reducing hallucinations!

CONVERSATION MEMORY
-------------------

We also maintain "memory" - a history of past Q&A exchanges.

Why? For natural follow-up questions:
- User: "What's the deadline?"
- AI: "December 31st"
- User: "Can it be extended?"  <- "it" refers to the deadline!
- AI: "Based on section 5, the deadline can be extended by..."

Without memory, the AI wouldn't know what "it" refers to.

PROMPT ENGINEERING
------------------

The system prompt is crucial. It tells Claude:
- What role to play (helpful assistant)
- What rules to follow (only use provided context)
- How to behave (be concise, cite sources)

A good prompt = better answers!

USAGE EXAMPLE
-------------
    from src.qa_chain import create_qa_chain, ask_question, format_response

    # Create the chain
    qa_chain = create_qa_chain(vector_store)

    # Ask questions
    result = ask_question(qa_chain, "What is the main topic?")
    print(format_response(result))

    # Follow-up (memory is maintained)
    result = ask_question(qa_chain, "Tell me more about that")
    print(format_response(result))
"""

import os
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import centralized configuration
from src.config import (
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    MEMORY_SIZE,
    TOP_K_RESULTS,
    MAX_RETRIES
)

# Load environment variables from .env file
# This is where your ANTHROPIC_API_KEY lives
load_dotenv()


# =============================================================================
# THE SYSTEM PROMPT - This is critical for good responses!
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

IMPORTANT RULES:
1. ONLY answer based on the provided context below
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the document to answer that question."
3. Be concise but thorough - give complete answers without unnecessary padding
4. When relevant, mention which part of the document your answer comes from
5. Don't make up information that isn't in the context

CONTEXT FROM THE DOCUMENT:
{context}

CONVERSATION HISTORY (for follow-up questions):
{chat_history}

Remember: Your knowledge is limited to the document context above. If asked about something not in the context, politely explain that the information isn't available in the provided document.
"""

# WHY THIS PROMPT WORKS:
# - Clear role definition ("helpful assistant")
# - Explicit rules (prevents hallucination)
# - Context placeholder (filled with retrieved chunks)
# - History placeholder (enables follow-ups)
# - Reminder at the end (reinforces boundaries)


class QAChain:
    """
    The main question-answering chain that connects all components.

    This class:
    1. Holds the vector store for retrieval
    2. Manages conversation memory
    3. Sends prompts to Claude
    4. Handles API errors gracefully

    Architecture:
        User Question
             |
             v
        [Retriever] ---> Get relevant chunks
             |
             v
        [Prompt Builder] ---> Combine context + history + question
             |
             v
        [Claude API] ---> Generate answer
             |
             v
        [Memory] ---> Store for follow-ups
             |
             v
        Answer + Sources
    """

    def __init__(self, vector_store, model_name: str = MODEL_NAME):
        """
        Initialize the QA chain with a vector store.

        Args:
            vector_store: ChromaDB vector store with embedded documents
            model_name: Which Claude model to use (from config by default)

        What Gets Set Up:
            - Retriever: Configured to fetch top K chunks
            - LLM: Claude client with API key
            - Prompt: Template for structuring requests
            - Memory: Empty list to store conversation history
        """
        # Store reference to vector store
        self.vector_store = vector_store

        # Create a retriever - this wraps the vector store for easy querying
        # search_kwargs={"k": 4} means "return 4 most relevant chunks"
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": TOP_K_RESULTS}
        )

        # Initialize empty conversation memory
        # Format: [(question1, answer1), (question2, answer2), ...]
        self.chat_history = []

        # Create the Claude LLM client
        # This handles all communication with Anthropic's API
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=TEMPERATURE,  # 0 = deterministic (best for Q&A)
            max_tokens=MAX_TOKENS,    # Maximum response length
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")  # From .env file
        )

        # Create the prompt template
        # ChatPromptTemplate structures the conversation for the API
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),  # Instructions for Claude
            ("human", "{question}")      # The user's question
        ])

        # Create the processing chain using LCEL (LangChain Expression Language)
        # This reads as: prompt -> send to LLM -> parse output as string
        # The | operator chains these steps together
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_chat_history(self) -> str:
        """
        Format conversation history for inclusion in the prompt.

        Returns:
            Formatted string of recent Q&A exchanges

        Why Limit History?
            - Each exchange adds tokens (costs money)
            - Too much history can confuse the model
            - Recent context is usually most relevant

        Example Output:
            "Human: What is the deadline?
             Assistant: The deadline is December 31st.
             Human: Can it be extended?
             Assistant: Yes, extensions are possible with written notice."
        """
        if not self.chat_history:
            return "No previous conversation."

        # Only keep the last N exchanges (from config)
        recent_history = self.chat_history[-MEMORY_SIZE:]

        # Format as Human/Assistant pairs
        formatted = []
        for question, answer in recent_history:
            formatted.append(f"Human: {question}")
            formatted.append(f"Assistant: {answer}")

        return "\n".join(formatted)

    def _format_context(self, docs: list) -> str:
        """
        Format retrieved documents into a context string for the prompt.

        Args:
            docs: List of Document objects from similarity search

        Returns:
            Formatted string with chunk numbers and content

        Example Output:
            "[Chunk 1, Page 3]:
             The payment terms are net-30 from invoice date...

             [Chunk 2, Page 5]:
             Late payments incur a 1.5% monthly fee..."
        """
        context_parts = []

        for i, doc in enumerate(docs, 1):  # Start numbering at 1
            # Get page number from metadata (if available)
            page = doc.metadata.get("page", "Unknown")

            # Format: [Chunk N, Page M]: content
            context_parts.append(
                f"[Chunk {i}, Page {page}]:\n{doc.page_content}"
            )

        # Join with double newlines for readability
        return "\n\n".join(context_parts)

    def _distance_to_confidence(self, distance: float) -> float:
        """
        Convert ChromaDB distance score to a confidence percentage.

        ChromaDB uses L2 (Euclidean) distance by default:
        - 0.0 = identical (100% confidence)
        - Higher values = less similar

        We convert this to a percentage where:
        - 100% = perfect match
        - 0% = completely unrelated

        The formula: confidence = max(0, (2 - distance) / 2 * 100)
        This maps distance 0->100%, distance 2->0%

        Args:
            distance: The L2 distance from ChromaDB (typically 0 to 2+)

        Returns:
            Confidence percentage (0 to 100)
        """
        # Clamp to reasonable range and convert to percentage
        # Distance of 0 = 100%, Distance of 2+ = 0%
        confidence = max(0, (2 - distance) / 2) * 100
        return round(confidence, 1)

    def ask(self, question: str, max_retries: int = MAX_RETRIES) -> dict:
        """
        Ask a question and get an answer with sources and confidence scores.

        This is the main method - it orchestrates the entire RAG flow.

        Args:
            question: The user's natural language question
            max_retries: How many times to retry if API is busy

        Returns:
            Dictionary containing:
            - 'answer': The generated response from Claude
            - 'source_documents': The chunks used to generate the answer
            - 'confidence_scores': List of confidence percentages for each chunk

        The Flow:
            1. Retrieve relevant chunks WITH SCORES from vector store
            2. Format context and chat history
            3. Send to Claude (with retry logic)
            4. Save Q&A to memory
            5. Return answer, sources, and confidence scores

        Error Handling:
            - API overloaded: Retries with exponential backoff
            - Other errors: Raised to caller
        """
        # ===== STEP 1: RETRIEVE WITH SCORES =====
        # Use similarity_search_with_score to get relevance scores
        # Returns list of (Document, distance) tuples
        results_with_scores = self.vector_store.similarity_search_with_score(
            question,
            k=TOP_K_RESULTS
        )

        # Separate documents and scores
        docs = [doc for doc, score in results_with_scores]
        distances = [score for doc, score in results_with_scores]

        # Convert distances to confidence percentages
        confidence_scores = [self._distance_to_confidence(d) for d in distances]

        # ===== STEP 2: AUGMENT =====
        # Format the retrieved docs and history for the prompt
        context = self._format_context(docs)
        chat_history = self._format_chat_history()

        # ===== STEP 3: GENERATE =====
        # Send to Claude with retry logic for API errors
        answer = None
        for attempt in range(max_retries):
            try:
                # Invoke the chain: prompt -> LLM -> parse
                answer = self.chain.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "question": question
                })
                break  # Success! Exit retry loop

            except Exception as e:
                # Check if it's an overload error (API too busy)
                if "overloaded" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff: wait longer each retry
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"   API busy, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Not an overload error, or we're out of retries
                    raise

        # ===== STEP 4: REMEMBER =====
        # Save this exchange to memory for follow-up questions
        self.chat_history.append((question, answer))

        # ===== STEP 5: RETURN =====
        return {
            "answer": answer,
            "source_documents": docs,
            "confidence_scores": confidence_scores  # NEW: Include confidence scores
        }

    def clear_history(self):
        """
        Clear the conversation memory.

        Use when:
        - Starting a new topic
        - Memory becomes irrelevant
        - User requests a fresh start
        """
        self.chat_history = []
        print("Conversation history cleared.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_qa_chain(vector_store, model_name: str = MODEL_NAME) -> QAChain:
    """
    Factory function to create a QA chain.

    This is the recommended way to instantiate QAChain.

    Args:
        vector_store: ChromaDB vector store with documents
        model_name: Claude model to use

    Returns:
        Configured QAChain ready for questions

    Example:
        >>> qa_chain = create_qa_chain(vector_store)
        >>> result = qa_chain.ask("What is the main topic?")
    """
    return QAChain(vector_store, model_name)


def ask_question(qa_chain: QAChain, question: str) -> dict:
    """
    Convenience function to ask a question.

    Args:
        qa_chain: The QAChain instance
        question: User's question

    Returns:
        Dictionary with 'answer' and 'source_documents'
    """
    return qa_chain.ask(question)


def _sanitize_text(text: str) -> str:
    """
    Remove special characters that cause Windows console encoding issues.

    Windows command prompt can't display certain Unicode characters,
    which causes crashes. This function replaces them with ASCII equivalents.

    Args:
        text: Text that may contain special characters

    Returns:
        Text safe for Windows console output
    """
    # Common special characters and their ASCII replacements
    replacements = {
        '●': '*',    # Bullet point
        '•': '*',    # Another bullet
        '→': '->',   # Arrow
        '←': '<-',   # Left arrow
        '✓': '[x]',  # Checkmark
        '✗': '[ ]',  # X mark
        '—': '-',    # Em dash
        '–': '-',    # En dash
        '"': '"',    # Smart quotes
        '"': '"',
        ''': "'",    # Smart apostrophes
        ''': "'",
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove any remaining non-ASCII characters
    # 'replace' substitutes ? for unknown characters
    return text.encode('ascii', 'replace').decode('ascii')


def _get_filename_from_path(file_path: str) -> str:
    """
    Extract just the filename from a full path.

    Example:
        >>> _get_filename_from_path("C:/docs/report.pdf")
        'report.pdf'
    """
    # Handle both Windows and Unix paths
    return os.path.basename(file_path) if file_path else "Unknown"


def _format_source_preview(text: str, max_length: int = 300) -> str:
    """
    Create a clean, readable preview of source text.

    This function:
    1. Cleans up whitespace (removes excessive newlines/spaces)
    2. Truncates to max_length characters
    3. Tries to end at a word boundary
    4. Wraps text for better readability

    Args:
        text: The raw source text
        max_length: Maximum characters to show

    Returns:
        Cleaned and truncated text preview
    """
    # Clean up whitespace - replace multiple spaces/newlines with single space
    cleaned = ' '.join(text.split())

    # Truncate if needed
    if len(cleaned) > max_length:
        # Try to cut at a word boundary
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # Only if we don't lose too much
            truncated = truncated[:last_space]
        cleaned = truncated + "..."

    return cleaned


def _wrap_text(text: str, width: int = 70, indent: str = "    ") -> str:
    """
    Wrap text to a specified width with indentation.

    Args:
        text: Text to wrap
        width: Maximum line width
        indent: String to prepend to each line

    Returns:
        Wrapped text with indentation
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(indent + ' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(indent + ' '.join(current_line))

    return '\n'.join(lines)


def format_response(result: dict) -> str:
    """
    Format the QA result for nice terminal display.

    Takes the raw result dictionary and creates a human-readable
    output with the answer, source references, and confidence scores.

    Args:
        result: Dictionary from ask_question()

    Returns:
        Formatted string ready for printing

    Output Format:
        [Answer text]

        --- Sources (4 chunks retrieved) ---

        [1] roman_empire.txt | Page 0 | 92.5% match
            "The Roman Empire was one of the largest empires in ancient
            history. At its peak under Emperor Trajan, it covered..."

    Confidence Score Interpretation:
        90-100%: Highly relevant - strong match
        70-89%:  Good relevance - likely useful
        50-69%:  Moderate relevance - may be tangential
        Below 50%: Low relevance - weak connection
    """
    answer = result.get("answer", "No answer found.")
    sources = result.get("source_documents", [])
    confidence_scores = result.get("confidence_scores", [])

    # Sanitize answer for Windows console
    output = f"\n{_sanitize_text(answer)}\n"

    # Add source references with enhanced formatting
    if sources:
        output += f"\n--- Sources ({len(sources)} chunks retrieved) ---\n"

        for i, doc in enumerate(sources, 1):
            # Extract metadata
            source_path = doc.metadata.get("source", "Unknown")
            filename = _get_filename_from_path(source_path)
            page = doc.metadata.get("page", None)

            # Get confidence score
            if i <= len(confidence_scores):
                confidence = confidence_scores[i - 1]
                confidence_str = f"{confidence}%"
            else:
                confidence_str = "N/A"

            # Build the header line
            # For PDFs: show page number
            # For TXT/MD: no page number (they don't have pages)
            if page is not None:
                header = f"[{i}] {filename} | Page {page} | {confidence_str} match"
            else:
                header = f"[{i}] {filename} | {confidence_str} match"

            # Format the preview text
            preview_text = _format_source_preview(doc.page_content, max_length=300)
            preview_text = _sanitize_text(preview_text)
            wrapped_preview = _wrap_text(f'"{preview_text}"', width=65, indent="    ")

            output += f"\n{header}\n{wrapped_preview}\n"

    return output


# =============================================================================
# EDUCATIONAL NOTES FOR DEVELOPERS
# =============================================================================
"""
PROMPT ENGINEERING TIPS
-----------------------

The system prompt is one of the most important parts of a RAG system.
Here's what makes a good one:

1. ROLE: Tell the AI who it is
   "You are a helpful assistant..." vs "You are a legal expert..."

2. RULES: Be explicit about what it should/shouldn't do
   - "Only use the provided context"
   - "If you don't know, say so"
   - "Don't make up information"

3. FORMAT: Describe how answers should be structured
   - "Be concise"
   - "Use bullet points when listing items"
   - "Cite the relevant section"

4. CONTEXT: Provide the retrieved information clearly
   - Use clear delimiters
   - Include metadata (page numbers)
   - Order by relevance

5. EXAMPLES: For complex tasks, show examples of good answers
   (Not done here, but powerful for specific use cases)

ALTERNATIVE ARCHITECTURES
-------------------------

1. Simple Retrieval (what we use):
   Question -> Retrieve -> Generate
   Pros: Simple, fast, cheap
   Cons: May not find all relevant info

2. Multi-Query Retrieval:
   Question -> Generate multiple query variants -> Retrieve all -> Merge -> Generate
   Pros: Better recall
   Cons: More API calls, slower

3. Self-RAG:
   Question -> Retrieve -> Generate -> Evaluate -> Maybe retrieve more
   Pros: Higher quality
   Cons: More complex, expensive

4. Agentic RAG:
   Question -> Agent decides what to do -> Maybe search, maybe calculate -> Answer
   Pros: Very flexible
   Cons: Complex, harder to control

DEBUGGING TIPS
--------------

1. Print retrieved chunks to see what the LLM sees:
   >>> result = qa_chain.ask("my question")
   >>> for doc in result['source_documents']:
   ...     print(doc.page_content[:100])

2. Check if the answer is grounded in context:
   - If answer includes info not in sources, something is wrong
   - Might need to improve retrieval or prompt

3. Test with simple questions first:
   - Ask something you KNOW is in the document
   - Verify it retrieves the right chunks

4. Experiment with temperature:
   - 0.0: Deterministic (best for factual Q&A)
   - 0.3-0.5: Slightly creative (for summaries)
   - 0.7+: More creative (usually not good for RAG)

COST OPTIMIZATION
-----------------

Every API call costs money. Here's how to optimize:

1. Use smaller models for simple questions:
   - Haiku for straightforward lookups
   - Sonnet for complex reasoning

2. Reduce context size:
   - Lower TOP_K_RESULTS if answers are good
   - Smaller chunks = less tokens

3. Cache common questions:
   - Store answers to frequently asked questions
   - Return cached answer instead of calling API

4. Batch questions:
   - If user has multiple questions, process together
"""
