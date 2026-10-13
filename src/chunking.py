import logging
from typing import Any, Dict, List, Sequence, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.contracts import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class ChunkerError(Exception):
    """Base exception for chunker operations."""


class InvalidChunkConfigError(ChunkerError):
    """Raised when chunker parameters fail validation."""


class ChunkingProcessError(ChunkerError):
    """Raised when text splitting encounters an operational error."""


# ---------------------------------------------------------------------------
# Configuration & Validation
# ---------------------------------------------------------------------------
class ChunkerConfig(BaseModel):
    """Configuration schema for text splitting parameters."""

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=500, gt=0, description="Chunk target size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between consecutive chunks")

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkerConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )
        return self


# ---------------------------------------------------------------------------
# Core Text Chunker
# ---------------------------------------------------------------------------
class TextChunker:
    """Splits Document objects into chunked Document instances while preserving metadata."""

    def __init__(
        self,
        config: Union[ChunkerConfig, None] = None,
        chunk_size: Union[int, None] = None,
        chunk_overlap: Union[int, None] = None,
    ) -> None:
        """Initializes the TextChunker.

        Can accept a `ChunkerConfig` instance or individual parameters.
        """
        try:
            if config is not None:
                self.config = config
            else:
                kwargs: Dict[str, Any] = {}
                if chunk_size is not None:
                    kwargs["chunk_size"] = chunk_size
                if chunk_overlap is not None:
                    kwargs["chunk_overlap"] = chunk_overlap
                self.config = ChunkerConfig(**kwargs)
        except ValidationError as exc:
            logger.error("Invalid chunker configuration: %s", exc)
            raise InvalidChunkConfigError(f"Configuration error: {exc}") from exc

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            
        )

        logger.debug(
            "Initialized TextChunker (chunk_size=%d, chunk_overlap=%d)",
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

    def chunk_document(self, document: Document) -> List[Document]:
        """Splits a single Document into smaller Document chunks with updated metadata."""
        if not isinstance(document, Document):
            raise ChunkingProcessError(f"Expected input of type Document, got {type(document)}")

        try:
            # LangChain splitter returns raw sub-documents with partial metadata
            raw_chunks = self._splitter.split_text(document.page_content)
            chunk_docs: List[Document] = []

            for index, chunk_text in enumerate(raw_chunks):
                # Preserve original metadata and inject chunk-specific attributes
                chunk_metadata = {
                    **document.metadata,
                    "chunk_index": index,
                    "total_chunks_in_page": len(raw_chunks),
                }

                chunk_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

            logger.debug(
                "Split document '%s' (page %s) into %d chunks",
                document.metadata.get("file_name", "unknown"),
                document.metadata.get("page_number", "N/A"),
                len(chunk_docs),
            )
            return chunk_docs

        except Exception as exc:
            logger.exception("Failed to chunk document: %s", exc)
            raise ChunkingProcessError(f"Error during chunking process: {exc}") from exc

    def chunk_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Processes a list of Document objects in batch."""
        total_input_docs = len(documents)
        logger.info("Starting batch chunking for %d input document(s)...", total_input_docs)

        all_chunks: List[Document] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            "Batch chunking complete: %d page document(s) -> %d total chunks",
            total_input_docs,
            len(all_chunks),
        )
        return all_chunks