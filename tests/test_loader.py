from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pypdf.errors import EmptyFileError, PyPdfError

from src.loader import (
    Document,
    DocumentNotFoundError,
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)


# ---------------------------------------------------------------------------
# 1. File Validation & Error Tests
# ---------------------------------------------------------------------------
def test_load_document_not_found():
    """Verify that a non-existent file raises DocumentNotFoundError."""
    non_existent = Path("non_existent_file_9999.txt")
    with pytest.raises(DocumentNotFoundError) as exc_info:
        load_document(non_existent)
    assert "File not found" in str(exc_info.value)


def test_load_empty_file_zero_bytes(tmp_path: Path):
    """Verify that a 0-byte file raises EmptyDocumentError."""
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")

    with pytest.raises(EmptyDocumentError) as exc_info:
        load_document(p)
    assert "is empty (0 bytes)" in str(exc_info.value)


def test_load_unsupported_file_type(tmp_path: Path):
    """Verify that unsupported extensions raise UnsupportedFileTypeError."""
    p = tmp_path / "document.docx"
    p.write_text("Some content", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        load_document(p)
    assert "not supported" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Text File Loader Tests
# ---------------------------------------------------------------------------
def test_load_txt_success(tmp_path: Path):
    """Verify that a valid text file loads correctly with proper metadata."""
    p = tmp_path / "sample.txt"
    content = "Hello, RAG world! This is a test text file."
    p.write_text(content, encoding="utf-8")

    docs = load_document(p)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == content
    assert docs[0].metadata["file_name"] == "sample.txt"
    assert docs[0].metadata["extension"] == ".txt"
    assert docs[0].metadata["total_pages"] == 1
    assert docs[0].metadata["page_number"] == 1
    assert "byte_size" in docs[0].metadata


@pytest.mark.parametrize("whitespace_content", ["   ", "\n\t   \n"])
def test_load_txt_empty_or_whitespace_raises_error(tmp_path: Path, whitespace_content: str):
    """Verify that empty or whitespace-only text files raise EmptyDocumentError."""
    p = tmp_path / "blank.txt"
    p.write_text(whitespace_content, encoding="utf-8")

    with pytest.raises(EmptyDocumentError) as exc_info:
        load_document(p)
    assert "contains no readable text" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. PDF File Loader Tests (Mocked)
# ---------------------------------------------------------------------------
@patch("src.loader.PyPDFLoader")
def test_load_pdf_success(mock_pdf_loader_class, tmp_path: Path):
    """Verify that a valid PDF file loads pages and structures metadata correctly."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-mock-bytes")  # dummy file so stat() checks pass

    # Mock PyPDFLoader instance and page outputs
    mock_instance = MagicMock()
    mock_page_1 = MagicMock()
    mock_page_1.page_content = "Page one content."
    mock_page_2 = MagicMock()
    mock_page_2.page_content = "Page two content."
    
    mock_instance.load.return_value = [mock_page_1, mock_page_2]
    mock_pdf_loader_class.return_value = mock_instance

    docs = load_document(p)

    assert len(docs) == 2
    assert docs[0].page_content == "Page one content."
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["total_pages"] == 2
    assert docs[0].metadata["file_name"] == "sample.pdf"

    assert docs[1].page_content == "Page two content."
    assert docs[1].metadata["page_number"] == 2
    assert docs[1].metadata["total_pages"] == 2


@patch("src.loader.PyPDFLoader")
def test_load_pdf_empty_file_error(mock_pdf_loader_class, tmp_path: Path):
    """Verify that an EmptyFileError from pypdf translates to EmptyDocumentError."""
    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%PDF-empty")

    mock_instance = MagicMock()
    mock_instance.load.side_effect = EmptyFileError("Empty PDF file")
    mock_pdf_loader_class.return_value = mock_instance

    with pytest.raises(EmptyDocumentError):
        load_document(p)


@patch("src.loader.PyPDFLoader")
def test_load_pdf_corrupted_error(mock_pdf_loader_class, tmp_path: Path):
    """Verify that a general PyPdfError translates to DocumentParsingError."""
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-broken")

    mock_instance = MagicMock()
    mock_instance.load.side_effect = PyPdfError("Corrupted file structure")
    mock_pdf_loader_class.return_value = mock_instance

    with pytest.raises(DocumentParsingError) as exc_info:
        load_document(p)
    assert "corrupted or unreadable" in str(exc_info.value)


@patch("src.loader.PyPDFLoader")
def test_load_pdf_no_extractable_text(mock_pdf_loader_class, tmp_path: Path):
    """Verify that a PDF with all blank pages raises EmptyDocumentError."""
    p = tmp_path / "blank.pdf"
    p.write_bytes(b"%PDF-blank")

    mock_instance = MagicMock()
    mock_page = MagicMock()
    mock_page.page_content = "   "  # whitespace only
    mock_instance.load.return_value = [mock_page]
    mock_pdf_loader_class.return_value = mock_instance

    with pytest.raises(EmptyDocumentError) as exc_info:
        load_document(p)
    assert "contains no extractable text" in str(exc_info.value)