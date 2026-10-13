"""Integration test suite for end-to-end RAG pipeline workflows."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.chunking import TextChunker
from src.embedding_service import EmbeddingService
from src.generation import GenerationError, GenerationService
from src.loader import Document
from src.rag_pipeline import InvalidPipelineInputError, PipelineExecutionError, RAGPipeline
from src.retriever import Retriever, RetrieverError
from src.schemas import FALLBACK_MESSAGE, IngestionSummary, RAGResponse, RetrievedSource
from src.vectordatabase import VectorStore


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for integration testing."""

    def __init__(self) -> None:
        self.indexed_docs = []
        self.indexed_embeddings = []

    def add_documents(self, documents, embeddings) -> None:
        self.indexed_docs.extend(documents)
        self.indexed_embeddings.extend(embeddings)


class FakeEmbeddingService(EmbeddingService):
    """Deterministic embedding service producing fixed-dimension vectors."""

    def embed_documents(self, documents):
        return [[0.1, 0.2, 0.3, 0.4] for _ in documents]

    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]


def test_end_to_end_ingestion_and_generation_flow(tmp_path):
    """1. Successful end-to-end flow: loading -> chunking -> embedding -> vector storage -> retrieval -> generation."""
    sample_file = tmp_path / "guide.pdf"
    sample_file.write_text("RAG systems combine vector retrieval with generative models.")

    chunker = TextChunker()
    embedding_service = FakeEmbeddingService()
    vector_store = InMemoryVectorStore()

    retriever = MagicMock(spec=Retriever)
    mock_doc = Document(
        page_content="RAG systems combine vector retrieval with generative models.",
        metadata={"file_name": "guide.pdf", "page_number": 2},
    )
    retriever.retrieve.return_value = [(mock_doc, 0.88)]

    gen_service = MagicMock(spec=GenerationService)
    gen_service.generate.return_value = "RAG integrates retrieval and generation for accurate answers."

    pipeline = RAGPipeline(
        retriever=retriever,
        generation_service=gen_service,
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunker=chunker,
    )

    # Ingestion test
    with patch("src.rag_pipeline.load_document") as mock_load:
        mock_load.return_value = [
            Document(page_content="RAG systems combine vector retrieval with generative models.", metadata={"file_name": "guide.pdf", "page_number": 2})
        ]
        summary = pipeline.ingest_documents([sample_file])

    assert isinstance(summary, IngestionSummary)
    assert summary.success is True
    assert summary.loaded_documents == 1
    assert summary.chunks_created > 0
    assert len(vector_store.indexed_docs) > 0

    # Run pipeline
    response = pipeline.run("What is RAG?")
    assert isinstance(response, RAGResponse)
    assert response.answer == "RAG integrates retrieval and generation for accurate answers."


def test_source_traceability():
    """2. Source traceability: the returned response contains the expected source metadata."""
    retriever = MagicMock(spec=Retriever)
    mock_doc = Document(
        page_content="Vector databases index dense embeddings.",
        metadata={"file_name": "architecture.pdf", "page_number": 5},
    )
    retriever.retrieve.return_value = [(mock_doc, 0.95)]

    gen_service = MagicMock(spec=GenerationService)
    gen_service.generate.return_value = "Vector databases index embeddings."

    pipeline = RAGPipeline(retriever=retriever, generation_service=gen_service)
    response = pipeline.run("Explain vector DBs")

    assert len(response.sources) == 1
    source = response.sources[0]
    assert isinstance(source, RetrievedSource)
    assert source.source == "architecture.pdf"
    assert source.page_number == 5
    assert source.score == 0.95
    assert source.content == "Vector databases index dense embeddings."


def test_no_relevant_documents_fallback():
    """3. No relevant documents: the pipeline returns the configured fallback response."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = []

    gen_service = MagicMock(spec=GenerationService)
    pipeline = RAGPipeline(retriever=retriever, generation_service=gen_service)

    response = pipeline.run("Unrelated query")
    assert isinstance(response, RAGResponse)
    assert response.answer == FALLBACK_MESSAGE
    assert response.sources == []
    gen_service.generate.assert_not_called()


def test_retrieval_failure_handling():
    """4. Retrieval failure: the pipeline handles the error according to its existing exception contract."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.side_effect = RetrieverError("Retriever connection error")

    gen_service = MagicMock(spec=GenerationService)
    pipeline = RAGPipeline(retriever=retriever, generation_service=gen_service)

    with pytest.raises(PipelineExecutionError) as exc_info:
        pipeline.run("Test query")
    assert "Retrieval failed" in str(exc_info.value)


def test_generation_failure_handling():
    """5. Generation failure: the pipeline handles generation errors correctly."""
    retriever = MagicMock(spec=Retriever)
    mock_doc = Document(page_content="Some text", metadata={"file_name": "doc.pdf"})
    retriever.retrieve.return_value = [(mock_doc, 0.8)]

    gen_service = MagicMock(spec=GenerationService)
    gen_service.generate.side_effect = GenerationError("LLM generation timeout")

    pipeline = RAGPipeline(retriever=retriever, generation_service=gen_service)

    with pytest.raises(PipelineExecutionError) as exc_info:
        pipeline.run("Test query")
    assert "Generation failed" in str(exc_info.value)


@pytest.mark.parametrize("invalid_query", ["", "   ", None, 123])
@pytest.mark.parametrize("invalid_top_k", [-1, 0, "two"])
def test_invalid_input_rejection(invalid_query, invalid_top_k):
    """6. Invalid input: empty queries and invalid top_k values are rejected."""
    pipeline = RAGPipeline(
        retriever=MagicMock(spec=Retriever),
        generation_service=MagicMock(spec=GenerationService),
    )

    # Test invalid query
    with pytest.raises(InvalidPipelineInputError):
        pipeline.run(invalid_query)  # type: ignore

    # Test invalid top_k
    with pytest.raises(InvalidPipelineInputError):
        pipeline.run("Valid query", top_k=invalid_top_k)  # type: ignore