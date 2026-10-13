from unittest.mock import MagicMock, patch
import pytest

from src.embedding_service import (
    EmbeddingConfig,
    EmbeddingModelError,
    EmbeddingService,
    InvalidEmbeddingConfigError,
    InvalidEmbeddingInputError,
)
from src.contracts import Document


# ---------------------------------------------------------------------------
# 1. Initialization & Configuration Tests
# ---------------------------------------------------------------------------
@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embedding_service_init_default(mock_hf_class):
    """Verify successful initialization with default configuration."""
    mock_hf_class.return_value = MagicMock()

    service = EmbeddingService()
    assert service.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    mock_hf_class.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embedding_service_init_custom_config(mock_hf_class):
    """Verify successful initialization with custom configuration parameters."""
    mock_hf_class.return_value = MagicMock()
    config = EmbeddingConfig(model_name="custom-model-id", model_kwargs={"device": "cuda"})

    service = EmbeddingService(config=config)
    assert service.config.model_name == "custom-model-id"
    assert service.config.model_kwargs == {"device": "cuda"}


@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embedding_service_model_load_failure(mock_hf_class):
    """Verify that exceptions during HuggingFaceEmbeddings loading wrap into EmbeddingModelError."""
    mock_hf_class.side_effect = Exception("Model download timeout")

    with pytest.raises(EmbeddingModelError) as exc_info:
        EmbeddingService()
    assert "Failed to load model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Query Embedding Tests (`embed_query`)
# ---------------------------------------------------------------------------
@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embed_query_success(mock_hf_class):
    """Verify successful vectorization of a valid query string."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_hf_class.return_value = mock_embedder

    service = EmbeddingService()
    vector = service.embed_query("What is RAG?")

    assert vector == [0.1, 0.2, 0.3, 0.4]
    mock_embedder.embed_query.assert_called_once_with("What is RAG?")


@pytest.mark.parametrize("invalid_query", [None, 123, 45.6, ["query"], {"q": "test"}])
def test_embed_query_invalid_types_rejected(invalid_query):
    """Verify that non-string queries raise InvalidEmbeddingInputError."""
    service = EmbeddingService.__new__(EmbeddingService)  # Bypass __init__ to test method validation directly
    with pytest.raises(InvalidEmbeddingInputError) as exc_info:
        service.embed_query(invalid_query)  # type: ignore
    assert "Expected input type 'str'" in str(exc_info.value)


@pytest.mark.parametrize("empty_query", ["", "   ", "\n\t   \n"])
def test_embed_query_empty_or_whitespace_rejected(empty_query):
    """Verify that empty or whitespace-only query strings raise InvalidEmbeddingInputError."""
    service = EmbeddingService.__new__(EmbeddingService)
    with pytest.raises(InvalidEmbeddingInputError) as exc_info:
        service.embed_query(empty_query)
    assert "Query string cannot be empty" in str(exc_info.value)


@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embed_query_model_failure_raises_error(mock_hf_class):
    """Verify that backend errors during query embedding raise EmbeddingModelError."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.side_effect = Exception("CUDA Out of Memory")
    mock_hf_class.return_value = mock_embedder

    service = EmbeddingService()
    with pytest.raises(EmbeddingModelError) as exc_info:
        service.embed_query("Valid query")
    assert "Failed to generate query embedding" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Document Batch Embedding Tests (`embed_documents`)
# ---------------------------------------------------------------------------
@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embed_documents_success(mock_hf_class):
    """Verify successful batch vectorization of valid Document instances."""
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_hf_class.return_value = mock_embedder

    docs = [
        Document(page_content="First document content.", metadata={"file_name": "a.txt", "page_number": 1}),
        Document(page_content="Second document content.", metadata={"file_name": "b.txt", "page_number": 1}),
    ]

    service = EmbeddingService()
    embeddings = service.embed_documents(docs)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]
    mock_embedder.embed_documents.assert_called_once_with([
        "First document content.",
        "Second document content.",
    ])


@pytest.mark.parametrize("invalid_sequence", [None, "not-a-sequence", 123, {"docs": []}])
def test_embed_documents_invalid_sequence_type_rejected(invalid_sequence):
    """Verify that passing non-Sequence objects raises InvalidEmbeddingInputError."""
    service = EmbeddingService.__new__(EmbeddingService)
    with pytest.raises(InvalidEmbeddingInputError):
        service.embed_documents(invalid_sequence)  # type: ignore


def test_embed_documents_empty_sequence_rejected():
    """Verify that passing an empty list/sequence raises InvalidEmbeddingInputError."""
    service = EmbeddingService.__new__(EmbeddingService)
    with pytest.raises(InvalidEmbeddingInputError) as exc_info:
        service.embed_documents([])
    assert "Document sequence cannot be empty" in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_element",
    [
        "just a string",
        123,
        None,
        Document(page_content="", metadata={}),  # Empty content
        Document(page_content="   ", metadata={}),  # Whitespace-only content
    ],
)
def test_embed_documents_invalid_elements_rejected(invalid_element):
    """Verify that non-Document items or Documents with empty content raise InvalidEmbeddingInputError."""
    service = EmbeddingService.__new__(EmbeddingService)
    
    if isinstance(invalid_element, Document):
        docs = [invalid_element]
    else:
        docs = [Document(page_content="Valid", metadata={}), invalid_element]  # type: ignore

    with pytest.raises(InvalidEmbeddingInputError):
        service.embed_documents(docs)  # type: ignore


@patch("src.embedding_service.HuggingFaceEmbeddings")
def test_embed_documents_model_failure_raises_error(mock_hf_class):
    """Verify that backend errors during batch document embedding raise EmbeddingModelError."""
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.side_effect = Exception("Runtime computation error")
    mock_hf_class.return_value = mock_embedder

    service = EmbeddingService()
    doc = Document(page_content="Valid content", metadata={})

    with pytest.raises(EmbeddingModelError) as exc_info:
        service.embed_documents([doc])
    assert "Failed to generate document embeddings" in str(exc_info.value)