import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from src.contracts import Document
from src.chunking import TextChunker
from src.embedding_service import EmbeddingService
from src.generation import GenerationError, GenerationService
from src.loader import DocumentLoaderError, load_document
from src.retriever import Retriever, RetrieverError
from src.schemas import FALLBACK_MESSAGE, IngestionSummary, RAGResponse, RetrievedSource
from src.vectordatabase import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline operations."""


class InvalidPipelineInputError(RAGPipelineError):
    """Raised when query or top_k fails validation."""


class PipelineExecutionError(RAGPipelineError):
    """Raised when ingestion, retrieval or generation fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class RAGPipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: Optional[int] = Field(default=None, gt=0, description="Overrides the retriever's default top_k")
    fallback_message: str = Field(
        default=FALLBACK_MESSAGE,
        description="Returned when no relevant chunks are found.",
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """Coordinates ingestion, retrieval and answer generation."""

    def __init__(
        self,
        retriever: Retriever,
        generation_service: GenerationService,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[TextChunker] = None,
        config: Optional[RAGPipelineConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.generation_service = generation_service
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunker = chunker
        self.config = config or RAGPipelineConfig()

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------
    def ingest_documents(self, file_paths: Sequence[Union[str, Path]]) -> IngestionSummary:
        """Loads, chunks, embeds and indexes files."""
        if self.chunker is None or self.embedding_service is None or self.vector_store is None:
            raise PipelineExecutionError(
                "Ingestion needs chunker, embedding_service and vector_store to be provided."
            )

        raw_docs: List[Document] = []
        failed_files: Dict[str, str] = {}
        primary_source = str(file_paths[0]) if file_paths else "Unknown"

        for path in file_paths:
            p = Path(path)
            try:
                raw_docs.extend(load_document(p))
            except DocumentLoaderError as exc:
                failed_files[str(p)] = str(exc)
                logger.warning("Skipping '%s': %s", p, exc)

        if not raw_docs:
            logger.warning("Ingestion finished with no loadable documents (%d failed).", len(failed_files))
            return IngestionSummary(
                source_path=primary_source,
                loaded_documents=0,
                chunks_created=0,
                chunks_indexed=0,
                success=False if failed_files else True,
            )

        try:
            chunks = self.chunker.chunk_documents(raw_docs)
            embeddings = self.embedding_service.embed_documents(chunks)
            self.vector_store.add_documents(chunks, embeddings)
        except Exception as exc:
            logger.exception("Ingestion failed")
            raise PipelineExecutionError(f"Ingestion failed: {exc}") from exc

        logger.info("Successfully ingested %d chunk(s).", len(chunks))

        return IngestionSummary(
            source_path=primary_source,
            loaded_documents=len(raw_docs),
            chunks_created=len(chunks),
            chunks_indexed=len(chunks),
            success=True,
        )

    # -----------------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------------
    def run(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """Runs retrieval + generation for a query."""
        self._validate_inputs(query, top_k)
        effective_top_k = top_k if top_k is not None else self.config.top_k

        try:
            retrieved = self.retriever.retrieve(query=query, top_k=effective_top_k)
        except RetrieverError as exc:
            logger.exception("Retrieval failed")
            raise PipelineExecutionError(f"Retrieval failed: {exc}") from exc

        if not retrieved:
            logger.info("No relevant chunks found; returning fallback answer.")
            return RAGResponse(
                question=query,
                answer=self.config.fallback_message,
                sources=[],
            )

        try:
            answer = self.generation_service.generate(
                query=query, documents=[doc for doc, _ in retrieved]
            )
        except GenerationError as exc:
            logger.exception("Generation failed")
            raise PipelineExecutionError(f"Generation failed: {exc}") from exc

        sources = []
        for doc, score in retrieved:
            raw_page = doc.metadata.get("page_number")
            page_number = (
                raw_page
                if isinstance(raw_page, int)
                else (int(raw_page) if isinstance(raw_page, str) and raw_page.isdigit() else None)
            )

            sources.append(
                RetrievedSource(
                    source=doc.metadata.get("file_name", "Unknown"),
                    page_number=page_number,
                    content=doc.page_content,
                    score=round(float(score), 4),
                )
            )

        return RAGResponse(
            question=query,
            answer=answer,
            sources=sources,
        )

    # -----------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(query: str, top_k: Optional[int]) -> None:
        if not isinstance(query, str) or not query.strip():
            raise InvalidPipelineInputError("Query must be a non-empty string.")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise InvalidPipelineInputError(f"top_k must be a positive integer, got {top_k!r}")