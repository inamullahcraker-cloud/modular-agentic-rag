import logging
from collections.abc import Sequence as CollectionsSequence
from typing import Any, Dict, List, Optional, Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.contracts import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions Hierarchy
# ---------------------------------------------------------------------------
class BaseEmbeddingError(Exception):
    """Base exception for all embedding operations."""


class InvalidEmbeddingConfigError(BaseEmbeddingError):
    """Raised when embedding configuration parameters fail validation."""


class InvalidEmbeddingInputError(BaseEmbeddingError):
    """Raised when provided input is invalid (empty string, wrong type, empty list)."""


class EmbeddingModelError(BaseEmbeddingError):
    """Raised when model loading or vector generation fails."""


# ---------------------------------------------------------------------------
# Configuration Schema
# ---------------------------------------------------------------------------
class EmbeddingConfig(BaseModel):
    """Configuration schema for HuggingFace embeddings."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model repository ID or local directory path",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"device": "cpu"},
        description="Device and runtime arguments passed to model constructor",
    )
    encode_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"normalize_embeddings": True},
        description="Encoding arguments (e.g., L2 normalization)",
    )


# ---------------------------------------------------------------------------
# Core Embedding Service
# ---------------------------------------------------------------------------
class EmbeddingService:
    """Service wrapper for LangChain HuggingFaceEmbeddings."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """Initializes the embedding service with configurable model settings."""
        try:
            self.config = config or EmbeddingConfig()
        except ValidationError as exc:
            logger.error("Invalid embedding configuration: %s", exc)
            raise InvalidEmbeddingConfigError(f"Configuration error: {exc}") from exc

        logger.info("Initializing HuggingFaceEmbeddings with model: '%s'", self.config.model_name)

        try:
            self._embedder = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs=self.config.model_kwargs,
                encode_kwargs=self.config.encode_kwargs,
            )
            logger.debug("Successfully loaded embedding model.")
        except Exception as exc:
            logger.exception("Failed to load embedding model '%s': %s", self.config.model_name, exc)
            raise EmbeddingModelError(
                f"Failed to load model '{self.config.model_name}': {exc}"
            ) from exc

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single query string."""
        if not isinstance(query, str):
            logger.error("Input query must be a string, got %s", type(query))
            raise InvalidEmbeddingInputError(
                f"Expected input type 'str', got '{type(query).__name__}'"
            )

        if not query.strip():
            logger.error("Attempted to embed an empty or whitespace query.")
            raise InvalidEmbeddingInputError("Query string cannot be empty or whitespace-only.")

        try:
            return self._embedder.embed_query(query)
        except Exception as exc:
            logger.exception("Error generating query embedding: %s", exc)
            raise EmbeddingModelError(f"Failed to generate query embedding: {exc}") from exc

    def embed_documents(self, documents: Sequence[Document]) -> List[List[float]]:
        """Generates embedding vectors for a sequence of Document objects."""
        # Use collections.abc.Sequence for runtime validation while excluding strings/bytes
        if not isinstance(documents, CollectionsSequence) or isinstance(documents, (str, bytes)):
            logger.error("Documents input must be a Sequence, got %s", type(documents))
            raise InvalidEmbeddingInputError(
                f"Expected a Sequence of Document objects, got '{type(documents).__name__}'"
            )

        if not documents:
            logger.error("Attempted to embed an empty document sequence.")
            raise InvalidEmbeddingInputError("Document sequence cannot be empty.")

        texts: List[str] = []
        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                logger.error("Element at index %d is not a Document instance.", idx)
                raise InvalidEmbeddingInputError(
                    f"Element at index {idx} must be a Document instance, got '{type(doc).__name__}'"
                )

            content = doc.page_content.strip()
            if not content:
                logger.error("Document at index %d contains empty page_content.", idx)
                raise InvalidEmbeddingInputError(
                    f"Document at index {idx} contains empty or whitespace-only page_content."
                )
            texts.append(content)

        try:
            logger.debug("Generating embeddings for %d document(s)...", len(texts))
            return self._embedder.embed_documents(texts)
        except Exception as exc:
            logger.exception("Error generating document embeddings: %s", exc)
            raise EmbeddingModelError(f"Failed to generate document embeddings: {exc}") from exc