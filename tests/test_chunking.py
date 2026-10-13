import pytest

from src.chunking import (
    ChunkerConfig,
    ChunkingProcessError,
    InvalidChunkConfigError,
    TextChunker,
)
from src.contracts import Document


# ---------------------------------------------------------------------------
# 1. Configuration & Validation Tests
# ---------------------------------------------------------------------------
def test_chunker_config_defaults():
    """Verify default values for ChunkerConfig."""
    config = ChunkerConfig()
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50


def test_chunker_config_valid_custom():
    """Verify valid custom parameters for ChunkerConfig."""
    config = ChunkerConfig(chunk_size=200, chunk_overlap=30)
    assert config.chunk_size == 200
    assert config.chunk_overlap == 30


@pytest.mark.parametrize(
    "size,overlap",
    [
        (100, 100),  # overlap equals chunk_size (invalid)
        (100, 120),  # overlap greater than chunk_size (invalid)
        (0, 10),     # zero chunk_size (invalid)
        (-50, 10),   # negative chunk_size (invalid)
        (100, -5),   # negative chunk_overlap (invalid)
    ],
)
def test_chunker_config_invalid_raises_error(size: int, overlap: int):
    """Verify that invalid sizes and overlaps raise ValueError."""
    with pytest.raises(ValueError):
        ChunkerConfig(chunk_size=size, chunk_overlap=overlap)


def test_text_chunker_init_with_kwargs():
    """Verify TextChunker initialization using direct keyword arguments."""
    chunker = TextChunker(chunk_size=300, chunk_overlap=25)
    assert chunker.config.chunk_size == 300
    assert chunker.config.chunk_overlap == 25


def test_text_chunker_init_invalid_config_raises_custom_error():
    """Verify that configuration validation errors translate to InvalidChunkConfigError."""
    with pytest.raises(InvalidChunkConfigError):
        TextChunker(chunk_size=50, chunk_overlap=100)


# ---------------------------------------------------------------------------
# 2. Single Document Chunking Tests
# ---------------------------------------------------------------------------
def test_chunk_single_document_success():
    """Verify that a long document splits into multiple chunks and preserves metadata."""
    chunker = TextChunker(chunk_size=40, chunk_overlap=10)
    doc = Document(
        page_content="This is a test string long enough to be split into multiple chunks by the text chunker utility.",
        metadata={"file_name": "test.txt", "page_number": 1, "custom_tag": "active"},
    )

    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Document)
        assert chunk.metadata["file_name"] == "test.txt"
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["custom_tag"] == "active"
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["total_chunks_in_page"] == len(chunks)
        assert len(chunk.page_content) <= 40


def test_chunk_short_document():
    """Verify that a short document (shorter than chunk size) results in a single chunk."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content="Short text document.",
        metadata={"file_name": "short.txt", "page_number": 1},
    )

    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text document."
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks_in_page"] == 1


def test_chunk_document_invalid_input_raises_error():
    """Verify that passing a non-Document object raises ChunkingProcessError."""
    chunker = TextChunker()
    with pytest.raises(ChunkingProcessError):
        chunker.chunk_document("not a document object")  # type: ignore


# ---------------------------------------------------------------------------
# 3. Batch Document Chunking Tests
# ---------------------------------------------------------------------------
def test_chunk_documents_batch_success():
    """Verify that batch processing multiple documents creates chunks for all."""
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    docs = [
        Document(page_content="First document text content here.", metadata={"file_name": "doc1.txt", "page_number": 1}),
        Document(page_content="Second document text content here.", metadata={"file_name": "doc2.txt", "page_number": 1}),
    ]

    all_chunks = chunker.chunk_documents(docs)
    
    assert len(all_chunks) >= 2
    file_names = {chunk.metadata["file_name"] for chunk in all_chunks}
    assert "doc1.txt" in file_names
    assert "doc2.txt" in file_names