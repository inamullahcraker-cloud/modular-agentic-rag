import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, ConfigDict, Field
from pypdf.errors import EmptyFileError, PyPdfError
from src.contracts import Document
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class DocumentLoaderError(Exception):
    """Base exception for all document loader errors."""


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when the specified file path does not exist or is not a file."""


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when the file extension is not supported."""


class DocumentParsingError(DocumentLoaderError):
    """Raised when an error occurs during file reading or parsing."""


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a document contains no printable text content."""





# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_document(file_path: Union[str, Path]) -> List[Document]:
    file = Path(file_path).resolve()

    # 1. Existence check
    if not file.exists() or not file.is_file():
        logger.error("File not found: %s", file)
        raise DocumentNotFoundError(f"File not found: '{file}'")

    # 2. Early 0-byte file check
    if file.stat().st_size == 0:
        logger.warning("Skipping 0-byte file: %s", file.name)
        raise EmptyDocumentError(f"File '{file.name}' is empty (0 bytes).")

    # 3. Extension validation
    extension = file.suffix.lower()
    if extension not in {".pdf", ".txt"}:
        logger.warning("Unsupported extension '%s' for file: %s", extension, file.name)
        raise UnsupportedFileTypeError(
            f"File extension '{extension}' is not supported."
        )

    # Only the file name is stored. The absolute server path would otherwise end up in
    # documents.json and in API responses (RetrievedSource.metadata).
    base_metadata: Dict[str, Any] = {
        "file_name": file.name,
        "extension": extension,
        "byte_size": file.stat().st_size,
    }

    logger.debug("Loading %s (%s)", file.name, extension)

    if extension == ".txt":
        return _load_txt(file, base_metadata)
    return _load_pdf(file, base_metadata)


# ---------------------------------------------------------------------------
# Internal Parsers
# ---------------------------------------------------------------------------
def _load_txt(file: Path, base_metadata: Dict[str, Any]) -> List[Document]:
    try:
        content = file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError, OSError) as exc:
        logger.error("Failed to read text file '%s': %s", file.name, exc)
        raise DocumentParsingError(f"Failed to read TXT file '{file.name}': {exc}") from exc

    # Validate content outside the reading try-block to avoid wrapping EmptyDocumentError
    cleaned_content = content.strip()
    if not cleaned_content:
        logger.warning("Skipping empty or whitespace-only file: %s", file.name)
        raise EmptyDocumentError(f"File '{file.name}' contains no readable text.")

    metadata = {**base_metadata, "total_pages": 1, "page_number": 1}
    return [Document(page_content=cleaned_content, metadata=metadata)]


def _load_pdf(file: Path, base_metadata: Dict[str, Any]) -> List[Document]:
    try:
        loader = PyPDFLoader(str(file))
        pdf_pages = loader.load()
    except EmptyFileError as exc:
        logger.warning("PDF file is empty: %s", file.name)
        raise EmptyDocumentError(f"PDF file '{file.name}' is an empty file.") from exc
    except PyPdfError as exc:
        logger.error("PyPDF parsing failed for '%s': %s", file.name, exc)
        raise DocumentParsingError(
            f"PDF file '{file.name}' is corrupted or unreadable."
        ) from exc
    except (PermissionError, OSError) as exc:
        logger.error("I/O error accessing PDF file '%s': %s", file.name, exc)
        raise DocumentParsingError(f"I/O error accessing PDF '{file.name}': {exc}") from exc

    total_pages = len(pdf_pages)
    documents: List[Document] = []

    for page_idx, page in enumerate(pdf_pages, start=1):
        text = page.page_content.strip()
        if not text:
            continue

        metadata = {
            **base_metadata,
            "page_number": page_idx,
            "total_pages": total_pages,
        }
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        logger.warning("PDF '%s' has no extractable text content.", file.name)
        raise EmptyDocumentError(f"PDF file '{file.name}' contains no extractable text.")

    logger.debug("Successfully loaded PDF '%s' (%d pages)", file.name, total_pages)
    return documents