"""Unit test suite for the FastAPI application (`src/api.py`)."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api import app
from src.contracts import IngestionSummary, RAGResponse

client = TestClient(app)


def test_serve_frontend():
    """Verify root endpoint serves the interactive HTML dashboard successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Modular RAG Assistant" in response.text
    assert "text/html" in response.headers["content-type"]


@patch("src.api.pipeline")
def test_ingest_documents_success(mock_pipeline):
    """Verify successful document upload and ingestion endpoint."""
    mock_pipeline.ingest_documents.return_value = IngestionSummary(
        total_documents_loaded=1,
        total_chunks_created=5,
        vector_dimension=384,
    )

    file_content = b"Sample document text content."
    response = client.post(
        "/api/ingest",
        files={"files": ("test.txt", file_content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_documents_loaded"] == 1
    assert data["total_chunks_created"] == 5
    mock_pipeline.ingest_documents.assert_called_once()


@patch("src.api.pipeline")
def test_ingest_documents_no_files(mock_pipeline):
    """Verify uploading empty or invalid files returns 422 Unprocessable Entity due to FastAPI validation."""
    response = client.post(
        "/api/ingest",
        files={"files": ("", b"", "application/octet-stream")},
    )
    assert response.status_code == 422


@patch("src.api.pipeline")
def test_ingest_documents_internal_error(mock_pipeline):
    """Verify internal exceptions during ingestion return 500 Internal Server Error."""
    mock_pipeline.ingest_documents.side_effect = Exception("Disk write error")

    response = client.post(
        "/api/ingest",
        files={"files": ("test.txt", b"content", "text/plain")},
    )
    assert response.status_code == 500
    assert "Disk write error" in response.json()["detail"]


@patch("src.api.pipeline")
def test_query_rag_success(mock_pipeline):
    """Verify successful RAG query endpoint execution with optional top_k parameter."""
    # Using dictionary initialization or model instantiation depending on schema fields
    mock_pipeline.run.return_value = {
        "query": "What is testing?",
        "answer": "This is a generated test answer.",
        "retrieved_sources": [
            {
                "file_name": "test.txt",
                "page_number": 1,
                "score": 0.95,
                "content": "Sample content chunk.",
            }
        ],
    }

    response = client.post(
        "/api/query",
        json={"query": "What is testing?", "top_k": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a generated test answer."
    assert len(data["retrieved_sources"]) == 1
    mock_pipeline.run.assert_called_once_with(query="What is testing?", top_k=2)


@patch("src.api.pipeline")
def test_query_rag_internal_error(mock_pipeline):
    """Verify internal exceptions during query execution return 500 Internal Server Error."""
    mock_pipeline.run.side_effect = Exception("LLM generation failure")

    response = client.post(
        "/api/query",
        json={"query": "What is testing?"},
    )
    assert response.status_code == 500
    assert "LLM generation failure" in response.json()["detail"]