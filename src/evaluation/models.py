"""Evaluation data models and Pydantic schemas for RAG pipeline testing."""

from typing import List, Optional, Set
from pydantic import BaseModel, Field


class RetrievalTestCase(BaseModel, frozen=True):
    """Represents a single test case for evaluating retrieval quality."""

    query: str = Field(..., description="Natural language query string.")
    relevant_doc_ids: Set[str] = Field(..., description="Set of known gold-standard/relevant document IDs.")
    retrieved_doc_ids: List[str] = Field(..., description="Ranked list of document IDs returned by the retriever.")


class RetrievalMetricsSummary(BaseModel, frozen=True):
    """Aggregated retrieval evaluation metrics across multiple test cases."""

    total_test_cases: int = Field(..., ge=0, description="Total number of test cases evaluated.")
    mean_precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Precision@K across test cases.")
    mean_recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Recall@K across test cases.")
    mean_hit_rate_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Hit Rate@K across test cases.")
    mean_mrr: float = Field(..., ge=0.0, le=1.0, description="Average Mean Reciprocal Rank across test cases.")
    k: int = Field(..., gt=0, description="Cutoff rank K used for evaluation.")


class EvaluationExample(BaseModel, frozen=True):
    """Represents an end-to-end evaluation example with query, relevant docs, and reference answer."""

    query: str = Field(..., description="The user's question.")
    relevant_document_ids: Set[str] = Field(..., description="IDs of documents considered relevant.")
    expected_answer: Optional[str] = Field(None, description="Reference answer for comparison.")


class SingleEvaluationResult(BaseModel, frozen=True):
    """Evaluation result for a single evaluation example."""

    query: str = Field(..., description="Evaluated query string.")
    success: bool = Field(..., description="Indicates whether pipeline execution succeeded.")
    generated_answer: Optional[str] = Field(None, description="Answer produced by the RAG pipeline.")
    expected_answer: Optional[str] = Field(None, description="Reference answer.")
    retrieved_doc_ids: List[str] = Field(default_factory=list, description="IDs of retrieved document chunks.")
    relevant_doc_ids: Set[str] = Field(default_factory=set, description="Ground truth relevant document IDs.")
    precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Precision@K score.")
    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Recall@K score.")
    hit_rate_at_k: float = Field(..., ge=0.0, le=1.0, description="Hit Rate@K score.")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank score.")
    error_message: Optional[str] = Field(None, description="Error message if pipeline execution failed.")


class EvaluationReport(BaseModel, frozen=True):
    """Comprehensive evaluation report across multiple test examples."""

    total_examples: int = Field(..., ge=0, description="Total number of evaluation examples.")
    successful_examples: int = Field(..., ge=0, description="Number of successfully evaluated examples.")
    failed_examples: int = Field(..., ge=0, description="Number of failed evaluation examples.")
    mean_precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Precision@K across successful examples.")
    mean_recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Recall@K across successful examples.")
    mean_hit_rate_at_k: float = Field(..., ge=0.0, le=1.0, description="Average Hit Rate@K across successful examples.")
    mean_mrr: float = Field(..., ge=0.0, le=1.0, description="Average MRR across successful examples.")
    k: int = Field(..., gt=0, description="Cutoff rank K used for evaluation.")
    results: List[SingleEvaluationResult] = Field(default_factory=list, description="Detailed per-example results.")