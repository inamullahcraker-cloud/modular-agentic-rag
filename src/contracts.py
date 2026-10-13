from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# Single definition, used by the prompt, the empty-answer fallback and the pipeline config.
FALLBACK_MESSAGE = "I cannot answer this question based on the provided documents."

# ---------------------------------------------------------------------------
# Data Representation
# ---------------------------------------------------------------------------
class Document(BaseModel):
    """Consistent document representation for RAG pipelines."""

    model_config = ConfigDict(frozen=True)

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestionSummary(BaseModel):
    """Summary of one ingestion run."""

    model_config = ConfigDict(frozen=True)

    total_documents_loaded: int = Field(..., ge=0, description="Pages/documents successfully loaded.")
    total_chunks_created: int = Field(..., ge=0, description="Chunks created after splitting.")
    vector_dimension: int = Field(..., ge=0, description="Embedding dimension (0 if nothing was ingested).")
    status: Literal["success", "partial", "no_documents"] = Field(
        default="success",
        description="success: all files loaded. partial: some files failed. no_documents: nothing loaded.",
    )
    source_files: List[str] = Field(default_factory=list, description="Files ingested successfully.")
    failed_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Files that were skipped, mapped to the reason.",
    )


class RetrievedSource(BaseModel):
    """A retrieved chunk with its source info and similarity score."""

    model_config = ConfigDict(frozen=True)

    content: str
    file_name: str
    page_number: Union[int, str]
    score: float = Field(..., description="Cosine similarity score.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Final response of a RAG query."""

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    retrieved_sources: List[RetrievedSource] = Field(default_factory=list)