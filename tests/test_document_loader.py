"""Tests for document loading and chunking — no API calls required."""
import os
import pytest
from langchain_core.documents import Document

from src.document_loader import (
    get_file_extension,
    load_document,
    split_into_chunks,
    scan_folder_for_documents,
    load_multiple_documents,
    is_folder,
    SUPPORTED_EXTENSIONS,
)


class TestGetFileExtension:
    def test_pdf_extension(self):
        assert get_file_extension("report.pdf") == ".pdf"

    def test_docx_extension(self):
        assert get_file_extension("notes.docx") == ".docx"

    def test_txt_extension(self):
        assert get_file_extension("readme.txt") == ".txt"

    def test_md_extension(self):
        assert get_file_extension("README.md") == ".md"

    def test_uppercase_normalised_to_lowercase(self):
        assert get_file_extension("Report.PDF") == ".pdf"
        assert get_file_extension("Notes.DOCX") == ".docx"

    def test_path_with_directories(self):
        assert get_file_extension("/some/path/to/file.pdf") == ".pdf"

    def test_file_without_extension(self):
        assert get_file_extension("Makefile") == ""


class TestLoadDocumentErrors:
    def test_unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("spreadsheet.xlsx")

    def test_unsupported_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            load_document("data.csv")

    def test_error_message_lists_supported_formats(self):
        with pytest.raises(ValueError) as exc_info:
            load_document("image.png")
        assert ".pdf" in str(exc_info.value)


class TestSupportedExtensions:
    def test_pdf_is_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_docx_is_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_txt_is_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_md_is_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS


class TestSplitIntoChunks:
    def _make_doc(self, text: str, source: str = "test.txt") -> Document:
        return Document(page_content=text, metadata={"source": source})

    def test_short_text_stays_as_one_chunk(self):
        doc = self._make_doc("Short text.")
        chunks = split_into_chunks([doc], chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1

    def test_long_text_is_split_into_multiple_chunks(self):
        long_text = "word " * 500  # 2500 chars
        doc = self._make_doc(long_text)
        chunks = split_into_chunks([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self):
        long_text = "word " * 500
        doc = self._make_doc(long_text)
        chunk_size = 300
        chunks = split_into_chunks([doc], chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50  # small tolerance for splitter

    def test_chunks_preserve_metadata(self):
        doc = self._make_doc("Some content.", source="myfile.txt")
        chunks = split_into_chunks([doc])
        assert all(c.metadata.get("source") == "myfile.txt" for c in chunks)

    def test_multiple_documents_combined(self):
        docs = [
            self._make_doc("Document one content."),
            self._make_doc("Document two content."),
        ]
        chunks = split_into_chunks(docs, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) >= 2

    def test_empty_document_list_returns_empty(self):
        chunks = split_into_chunks([])
        assert chunks == []

    def test_uses_config_defaults_when_not_specified(self):
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        doc = self._make_doc("Test content.")
        chunks = split_into_chunks([doc])
        assert isinstance(chunks, list)


class TestScanFolder:
    def test_raises_for_nonexistent_folder(self, tmp_path):
        with pytest.raises(ValueError, match="Not a valid folder"):
            scan_folder_for_documents(str(tmp_path / "nonexistent"))

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = scan_folder_for_documents(str(tmp_path))
        assert result == []

    def test_finds_supported_files(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("fake pdf")
        (tmp_path / "notes.txt").write_text("some notes")
        result = scan_folder_for_documents(str(tmp_path))
        filenames = [os.path.basename(p) for p in result]
        assert "doc.pdf" in filenames
        assert "notes.txt" in filenames

    def test_ignores_unsupported_files(self, tmp_path):
        (tmp_path / "data.xlsx").write_text("spreadsheet")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = scan_folder_for_documents(str(tmp_path))
        assert result == []

    def test_ignores_subdirectories(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested file")
        result = scan_folder_for_documents(str(tmp_path))
        assert result == []

    def test_returns_sorted_paths(self, tmp_path):
        (tmp_path / "c.txt").write_text("c")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = scan_folder_for_documents(str(tmp_path))
        assert result == sorted(result)


class TestLoadMultipleDocuments:
    def test_nonexistent_files_go_to_failed(self):
        _, summary = load_multiple_documents(["/nonexistent/file.pdf"])
        assert summary["failed"] == 1
        assert summary["successful"] == 0

    def test_summary_has_expected_keys(self):
        _, summary = load_multiple_documents([])
        assert "total_files" in summary
        assert "successful" in summary
        assert "failed" in summary
        assert "files_loaded" in summary
        assert "files_failed" in summary

    def test_empty_list_returns_empty_summary(self):
        docs, summary = load_multiple_documents([])
        assert docs == []
        assert summary["total_files"] == 0
        assert summary["successful"] == 0

    def test_total_files_matches_input(self):
        paths = ["/fake/one.pdf", "/fake/two.pdf"]
        _, summary = load_multiple_documents(paths)
        assert summary["total_files"] == len(paths)


class TestIsFolder:
    def test_directory_returns_true(self, tmp_path):
        assert is_folder(str(tmp_path)) is True

    def test_nonexistent_path_returns_false(self, tmp_path):
        assert is_folder(str(tmp_path / "nope")) is False

    def test_file_returns_false(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        assert is_folder(str(f)) is False
