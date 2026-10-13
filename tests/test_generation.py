"""Unit test suite for the GenerationService module."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from src.contracts import FALLBACK_MESSAGE
from src.loader import Document
from src.generation import (
    GenerationConfig,
    GenerationService,
    InvalidGenerationInputError,
    LLMGenerationError,
)


def test_generation_config_defaults():
    """Verify default configuration values."""
    config = GenerationConfig()
    assert config.model_name == "mistral"
    assert config.temperature == 0.0
    assert config.max_tokens == 512
    assert config.num_ctx == 4096
    assert config.timeout == 60.0


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"model_name": ""},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"max_tokens": 0},
        {"num_ctx": -10},
        {"timeout": 0.0},
    ],
)
def test_generation_config_invalid_fields(invalid_kwargs):
    """Verify invalid configuration parameters raise Pydantic validation errors."""
    with pytest.raises(ValidationError):
        GenerationConfig(**invalid_kwargs)


@patch("src.generation.ChatOllama")
def test_generation_service_init(mock_chat_ollama):
    """Verify successful initialization of GenerationService and ChatOllama client."""
    config = GenerationConfig(model_name="llama3", temperature=0.2)
    service = GenerationService(config=config)

    assert service.config.model_name == "llama3"
    assert service.config.temperature == 0.2
    mock_chat_ollama.assert_called_once_with(
        model="llama3",
        base_url=None,
        temperature=0.2,
        num_predict=512,
        num_ctx=4096,
        client_kwargs={"timeout": 60.0},
    )


@patch("src.generation.ChatOllama")
def test_generate_empty_documents_raises_error(mock_chat_ollama):
    """Verify that calling generate with an empty document sequence raises InvalidGenerationInputError."""
    service = GenerationService()
    with pytest.raises(InvalidGenerationInputError) as exc_info:
        service.generate("What is RAG?", [])
    assert "At least one context document is required" in str(exc_info.value)


@patch("src.generation.ChatOllama")
def test_generate_success_and_formatting(mock_chat_ollama):
    """Verify successful generation, prompt construction, and context formatting."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "RAG combines retrieval and generation."
    mock_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_instance

    service = GenerationService()
    doc = Document(
        page_content="RAG systems use vector search.",
        metadata={"file_name": "guide.pdf", "page_number": 3},
    )

    answer = service.generate("Explain RAG", [doc])

    assert answer == "RAG combines retrieval and generation."
    mock_instance.invoke.assert_called_once()

    # Verify message construction
    called_messages = mock_instance.invoke.call_args[0][0]
    assert len(called_messages) == 2
    system_msg = called_messages[0].content
    human_msg = called_messages[1].content

    assert "guide.pdf" in human_msg
    assert "Page: 3" in human_msg
    assert "RAG systems use vector search." in human_msg
    assert "Explain RAG" in human_msg
    assert FALLBACK_MESSAGE in system_msg


@patch("src.generation.ChatOllama")
def test_generate_empty_response_triggers_fallback(mock_chat_ollama):
    """Verify that an empty or whitespace LLM response content triggers the fallback message."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "   "  # Whitespace response
    mock_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_instance

    service = GenerationService()
    doc = Document(page_content="Content", metadata={"file_name": "doc.txt", "page_number": 1})

    answer = service.generate("Query?", [doc])
    assert answer == FALLBACK_MESSAGE


@patch("src.generation.ChatOllama")
def test_llm_invocation_failure_raises_generation_error(mock_chat_ollama):
    """Verify that network, timeout, or model errors during invoke wrap into LLMGenerationError."""
    mock_instance = MagicMock()
    mock_instance.invoke.side_effect = Exception("Connection timeout")
    mock_chat_ollama.return_value = mock_instance

    service = GenerationService()
    doc = Document(page_content="Content", metadata={"file_name": "doc.txt", "page_number": 1})

    with pytest.raises(LLMGenerationError) as exc_info:
        service.generate("Query?", [doc])
    assert "LLM call failed" in str(exc_info.value)