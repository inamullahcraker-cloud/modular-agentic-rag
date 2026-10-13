"""Unit test suite for the FAISS-backed VectorStore module."""

from pathlib import Path
import pytest
import numpy as np

from src.loader import Document
from src.vectordatabase import (
    VectorStore,
    document_id,
    VectorStoreValidationError,
    VectorStoreSearchError,
    VectorStorePersistenceError,
)


def test_document_id_determinism():
    """Verify that document_id generates consistent hashes for identical docs and different hashes for different docs."""
    doc1 = Document(page_content="Hello world", metadata={"file_name": "a.pdf", "page_number": 1, "chunk_index": 0})
    doc2 = Document(page_content="Hello world", metadata={"file_name": "a.pdf", "page_number": 1, "chunk_index": 0})
    doc3 = Document(page_content="Different text", metadata={"file_name": "a.pdf", "page_number": 1, "chunk_index": 0})

    assert document_id(doc1) == document_id(doc2)
    assert document_id(doc1) != document_id(doc3)


@pytest.mark.parametrize("invalid_dim", [0, -1, "384", None])
def test_vector_store_init_invalid_dimension(invalid_dim):
    """Verify initialization with non-positive or non-integer dimensions raises VectorStoreValidationError."""
    with pytest.raises(VectorStoreValidationError):
        VectorStore(dimension=invalid_dim)  # type: ignore


def test_vector_store_init_success():
    """Verify successful initialization of VectorStore."""
    store = VectorStore(dimension=4, embedding_model="test-model")
    assert store.dimension == 4
    assert store.embedding_model == "test-model"
    assert len(store) == 0


def test_filter_new_documents():
    """Verify filter_new correctly identifies unique documents and drops duplicates."""
    store = VectorStore(dimension=2)
    doc1 = Document(page_content="Doc 1", metadata={"file_name": "f1.pdf", "page_number": 1, "chunk_index": 0})
    doc2 = Document(page_content="Doc 2", metadata={"file_name": "f1.pdf", "page_number": 1, "chunk_index": 1})

    # Add doc1 to store first
    store.add_documents([doc1], [[1.0, 0.0]])

    # Filter a batch containing doc1 (duplicate), doc2 (new), and an in-batch duplicate of doc2
    batch = [doc1, doc2, doc2]
    fresh = store.filter_new(batch)

    assert len(fresh) == 1
    assert fresh[0].page_content == "Doc 2"


def test_add_documents_validation_and_success():
    """Verify validation checks during add_documents and successful addition/deduplication count."""
    store = VectorStore(dimension=3)
    doc1 = Document(page_content="A", metadata={"file_name": "doc.pdf", "page_number": 1, "chunk_index": 0})
    doc2 = Document(page_content="B", metadata={"file_name": "doc.pdf", "page_number": 1, "chunk_index": 1})

    # 1. Length mismatch
    with pytest.raises(VectorStoreValidationError):
        store.add_documents([doc1], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # 2. Embedding dimension mismatch
    with pytest.raises(VectorStoreValidationError):
        store.add_documents([doc1], [[1.0, 0.0]])  # Expected dim 3, got 2

    # 3. Successful addition of unique documents
    added_count = store.add_documents([doc1, doc2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert added_count == 2
    assert len(store) == 2

    # 4. Adding duplicate doc should be skipped
    added_again = store.add_documents([doc1], [[1.0, 0.0, 0.0]])
    assert added_again == 0
    assert len(store) == 2


def test_similarity_search_behavior(tmp_path):
    """Verify similarity search error handling, empty store returns, and correct ranking."""
    store = VectorStore(dimension=2)

    # Empty store search returns empty list
    assert store.similarity_search([1.0, 0.0], k=2) == []

    doc1 = Document(page_content="Target", metadata={"file_name": "target.txt", "page_number": 1, "chunk_index": 0})
    doc2 = Document(page_content="Other", metadata={"file_name": "other.txt", "page_number": 1, "chunk_index": 0})
    store.add_documents([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])

    # Dimension mismatch
    with pytest.raises(VectorStoreValidationError):
        store.similarity_search([1.0, 0.0, 0.0], k=1)

    # Invalid k
    with pytest.raises(VectorStoreSearchError):
        store.similarity_search([1.0, 0.0], k=0)

    # Successful search query matching doc1 closely
    results = store.similarity_search([0.9, 0.1], k=1)
    assert len(results) == 1
    retrieved_doc, score = results[0]
    assert retrieved_doc.page_content == "Target"
    assert score > 0.9


def test_persistence_save_and_load(tmp_path):
    """Verify saving vector store to disk and loading it back successfully with validation checks."""
    store_dir = tmp_path / "vector_store"
    store = VectorStore(dimension=3, embedding_model="mistral")

    doc = Document(page_content="Persistence test", metadata={"file_name": "p.pdf", "page_number": 1, "chunk_index": 0})
    store.add_documents([doc], [[0.5, 0.5, 0.7]])

    # Save store
    store.save(store_dir)
    assert (store_dir / VectorStore.INDEX_FILENAME).exists()
    assert (store_dir / VectorStore.DOCS_FILENAME).exists()
    assert (store_dir / VectorStore.MANIFEST_FILENAME).exists()

    # Load store successfully
    loaded_store = VectorStore.load(store_dir, expected_embedding_model="mistral")
    assert len(loaded_store) == 1
    assert loaded_store.dimension == 3
    assert loaded_store.embedding_model == "mistral"
    assert loaded_store._documents[0].page_content == "Persistence test"

    # Load with embedding model mismatch should raise error
    with pytest.raises(VectorStorePersistenceError):
        VectorStore.load(store_dir, expected_embedding_model="different-model")

    # Load from non-existent directory should raise error
    with pytest.raises(VectorStorePersistenceError):
        VectorStore.load(tmp_path / "non_existent")