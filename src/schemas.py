from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

FALLBACK_MESSAGE = "I cannot answer this question based on the provided documents."


class IngestionSummary(BaseModel):
    """Reports the outcome of ingesting a document or file."""

    model_config = ConfigDict(frozen=True)

    source_path: str = Field(..., description="Path or identifier of the ingested source file.")
    loaded_documents: int = Field(..., ge=0, description="Number of documents successfully loaded.")
    chunks_created: int = Field(..., ge=0, description="Total text chunks generated after splitting.")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks successfully indexed in the vector store.")
    success: bool = Field(..., description="Indicates whether the ingestion process completed successfully.")


class RetrievedSource(BaseModel):
    """Represents one retrieved document chunk."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source identifier or filename.")
    page_number: Optional[int] = Field(None, description="Page number if available, else None.")
    content: str = Field(..., description="Text content of the retrieved chunk.")
    score: float = Field(..., description="Similarity score matching retrieval ranking.")


class RAGResponse(BaseModel):
    """Represents the final answer to a user's question along with grounding sources."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., description="Original natural language user question.")
    answer: str = Field(..., description="Generated answer produced by the LLM generation service.")
    sources: List[RetrievedSource] = Field(default_factory=list, description="List of retrieved source chunks used as context.")