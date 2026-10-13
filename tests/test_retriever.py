"""Unit test suite for the Retriever component."""

from unittest.mock import MagicMock
import pytest

from src.embedding_service import BaseEmbeddingError, EmbeddingService
from src.loader import Document
from src.retriever import InvalidQueryError, RetrievalError, Retriever, RetrieverConfig
from src.vectordatabase import StoreError, VectorStore


def test_retriever_config_defaults():
    """Verify default configuration values."""
    config = RetrieverConfig()
    assert config.top_k == 4
    assert config.min_score == 0.3


def test_retriever_config_valid_custom():
    """Verify custom valid configuration values."""
    config = RetrieverConfig(top_k=2, min_score=0.7)
    assert config.top_k == 2
    assert config.min_score == 0.7


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"top_k": 0},
        {"top_k": -1},
        {"min_score": -1.5},
        {"min_score": 1.1},
    ],
)
def test_retriever_config_invalid_raises_error(invalid_kwargs):
    """Verify invalid configuration values raise validation errors."""
    with pytest.raises(Exception):
        RetrieverConfig(**invalid_kwargs)


def test_retriever_initialization_success():
    """Verify successful retriever initialization."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 5

    retriever = Retriever(embedding_service=mock_embedding, vector_store=mock_store)
    assert retriever.embedding_service == mock_embedding
    assert retriever.vector_store == mock_store
    assert retriever.config.top_k == 4


def test_retrieve_empty_vector_store():
    """Verify retrieving from an empty vector store returns an empty list immediately without calling embedding service."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 0

    retriever = Retriever(embedding_service=mock_embedding, vector_store=mock_store)
    results = retriever.retrieve("What is RAG?")

    assert results == []
    mock_embedding.embed_query.assert_not_called()


def test_retrieve_success_with_score_filtering():
    """Verify successful retrieval, score sorting, and min_score filtering."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 3

    doc1 = Document(page_content="High score doc", metadata={"file_name": "a.pdf", "page_number": 1})
    doc2 = Document(page_content="Medium score doc", metadata={"file_name": "b.pdf", "page_number": 2})
    doc3 = Document(page_content="Low score doc (filtered out)", metadata={"file_name": "c.pdf", "page_number": 3})

    # Vector store returns results sorted by score descending
    mock_store.similarity_search.return_value = [
        (doc1, 0.85),
        (doc2, 0.45),
        (doc3, 0.15),  # Below default min_score=0.3
    ]

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_store,
        config=RetrieverConfig(top_k=3, min_score=0.3),
    )

    results = retriever.retrieve("Explain vector embeddings", top_k=3)

    mock_embedding.embed_query.assert_called_once_with("Explain vector embeddings")
    mock_store.similarity_search.assert_called_once_with([0.1, 0.2, 0.3], k=3)

    # doc3 should be filtered out because 0.15 < 0.3
    assert len(results) == 2
    assert results[0] == (doc1, 0.85)
    assert results[1] == (doc2, 0.45)


@pytest.mark.parametrize("invalid_top_k", [0, -1])
def test_retrieve_invalid_top_k_raises_error(invalid_top_k):
    """Verify that non-positive top_k overrides raise InvalidQueryError."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 2

    retriever = Retriever(embedding_service=mock_embedding, vector_store=mock_store)

    with pytest.raises(InvalidQueryError):
        retriever.retrieve("Valid query", top_k=invalid_top_k)


def test_embedding_service_failure_raises_retrieval_error():
    """Verify that BaseEmbeddingError from EmbeddingService wraps into RetrievalError."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_embedding.embed_query.side_effect = BaseEmbeddingError("Model timeout")

    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 5

    retriever = Retriever(embedding_service=mock_embedding, vector_store=mock_store)

    with pytest.raises(RetrievalError) as exc_info:
        retriever.retrieve("Query text")
    assert "Retrieval failed" in str(exc_info.value)


def test_vector_store_search_failure_raises_retrieval_error():
    """Verify that StoreError from VectorStore wraps into RetrievalError."""
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_embedding.embed_query.return_value = [0.1, 0.2]

    mock_store = MagicMock(spec=VectorStore)
    mock_store.__len__.return_value = 5
    mock_store.similarity_search.side_effect = StoreError("Index corruption")

    retriever = Retriever(embedding_service=mock_embedding, vector_store=mock_store)

    with pytest.raises(RetrievalError) as exc_info:
        retriever.retrieve("Query text")
    assert "Retrieval failed" in str(exc_info.value)