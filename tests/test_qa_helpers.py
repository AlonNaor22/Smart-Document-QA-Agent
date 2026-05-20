"""Tests for pure helper functions in qa_chain — no API calls required."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.qa_chain import (
    _sanitize_text,
    _get_filename_from_path,
    _format_source_preview,
    _wrap_text,
    QAChain,
)


class TestSanitizeText:
    def test_bullet_point_replaced(self):
        assert _sanitize_text("● item") == "* item"

    def test_arrow_replaced(self):
        assert _sanitize_text("step → next") == "step -> next"

    def test_em_dash_replaced(self):
        assert _sanitize_text("one — two") == "one - two"

    def test_smart_quotes_replaced(self):
        text = chr(0x201c) + "hello" + chr(0x201d)  # U+201C/U+201D
        result = _sanitize_text(text)
        assert "hello" in result
        assert result.isascii()

    def test_plain_ascii_unchanged(self):
        assert _sanitize_text("Hello, world!") == "Hello, world!"

    def test_non_ascii_becomes_question_mark(self):
        result = _sanitize_text("café")
        assert "?" in result

    def test_empty_string_returns_empty(self):
        assert _sanitize_text("") == ""


class TestGetFilenameFromPath:
    def test_simple_filename(self):
        assert _get_filename_from_path("report.pdf") == "report.pdf"

    def test_unix_path(self):
        assert _get_filename_from_path("/home/user/docs/report.pdf") == "report.pdf"

    def test_windows_path(self):
        assert _get_filename_from_path("C:\\Users\\user\\docs\\report.pdf") == "report.pdf"

    def test_empty_string_returns_unknown(self):
        # Empty string is falsy — function treats it like None and returns "Unknown"
        assert _get_filename_from_path("") == "Unknown"

    def test_none_returns_unknown(self):
        assert _get_filename_from_path(None) == "Unknown"


class TestFormatSourcePreview:
    def test_short_text_unchanged(self):
        text = "Short text here."
        result = _format_source_preview(text, max_length=300)
        assert "Short text here." in result

    def test_long_text_truncated(self):
        long_text = "word " * 100  # 500 chars
        result = _format_source_preview(long_text, max_length=50)
        assert len(result) <= 60  # allows for "..." suffix

    def test_truncated_text_ends_with_ellipsis(self):
        long_text = "a" * 500
        result = _format_source_preview(long_text, max_length=100)
        assert result.endswith("...")

    def test_whitespace_normalized(self):
        text = "hello   \n\n  world"
        result = _format_source_preview(text)
        assert "  " not in result  # no double spaces
        assert "\n" not in result

    def test_empty_string_returns_empty(self):
        assert _format_source_preview("") == ""


class TestWrapText:
    def test_short_text_fits_on_one_line(self):
        result = _wrap_text("Short text.", width=80)
        assert "\n" not in result

    def test_long_text_is_wrapped(self):
        long_text = "word " * 30  # well over 70 chars
        result = _wrap_text(long_text, width=30)
        assert "\n" in result

    def test_indent_applied(self):
        result = _wrap_text("some text", indent=">>> ")
        assert result.startswith(">>> ")

    def test_empty_string_returns_empty_or_indent(self):
        result = _wrap_text("", indent="  ")
        assert result == ""


class TestQAChainPureMethods:
    """Test QAChain methods that don't need the LLM or vector store."""

    @pytest.fixture
    def chain(self):
        obj = MagicMock(spec=QAChain)
        obj.chat_history = []
        obj._format_chat_history = QAChain._format_chat_history.__get__(obj)
        obj._format_context = QAChain._format_context.__get__(obj)
        obj._distance_to_confidence = QAChain._distance_to_confidence.__get__(obj)
        return obj

    class TestFormatChatHistory:
        @pytest.fixture
        def chain(self):
            obj = MagicMock(spec=QAChain)
            obj.chat_history = []
            obj._format_chat_history = QAChain._format_chat_history.__get__(obj)
            return obj

        def test_empty_history_returns_no_conversation(self, chain):
            result = chain._format_chat_history()
            assert "No previous conversation" in result

        def test_one_exchange_formatted_correctly(self, chain):
            chain.chat_history = [("What is X?", "X is Y.")]
            result = chain._format_chat_history()
            assert "Human: What is X?" in result
            assert "Assistant: X is Y." in result

        def test_multiple_exchanges_all_present(self, chain):
            chain.chat_history = [
                ("Q1", "A1"),
                ("Q2", "A2"),
            ]
            result = chain._format_chat_history()
            assert "Q1" in result and "A1" in result
            assert "Q2" in result and "A2" in result

    class TestFormatContext:
        @pytest.fixture
        def chain(self):
            obj = MagicMock(spec=QAChain)
            obj._format_context = QAChain._format_context.__get__(obj)
            return obj

        def _make_doc(self, content: str, page: int = 0) -> Document:
            return Document(page_content=content, metadata={"page": page})

        def test_single_doc_formatted(self, chain):
            doc = self._make_doc("Important content.", page=3)
            result = chain._format_context([doc])
            assert "Important content." in result
            assert "Chunk 1" in result
            assert "Page 3" in result

        def test_multiple_docs_numbered(self, chain):
            docs = [self._make_doc("First."), self._make_doc("Second.")]
            result = chain._format_context(docs)
            assert "Chunk 1" in result
            assert "Chunk 2" in result

        def test_missing_page_shows_unknown(self, chain):
            doc = Document(page_content="content", metadata={})
            result = chain._format_context([doc])
            assert "Unknown" in result

    class TestDistanceToConfidence:
        @pytest.fixture
        def chain(self):
            obj = MagicMock(spec=QAChain)
            obj._distance_to_confidence = QAChain._distance_to_confidence.__get__(obj)
            return obj

        def test_zero_distance_is_full_confidence(self, chain):
            assert chain._distance_to_confidence(0.0) == 100.0

        def test_distance_two_is_zero_confidence(self, chain):
            assert chain._distance_to_confidence(2.0) == 0.0

        def test_distance_one_is_half_confidence(self, chain):
            assert chain._distance_to_confidence(1.0) == 50.0

        def test_large_distance_clamps_to_zero(self, chain):
            assert chain._distance_to_confidence(5.0) == 0.0

        def test_result_is_rounded(self, chain):
            result = chain._distance_to_confidence(0.333)
            assert result == round(result, 1)
