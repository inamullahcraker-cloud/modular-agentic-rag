"""Unit tests for the RAGEvaluator evaluation runner."""

import pytest

from src.evaluation.models import EvaluationExample, EvaluationReport
from src.evaluation.rag_evaluator import RAGEvaluator
from src.schemas import RAGResponse, RetrievedSource


class FakePipeline:
    """Fake RAG pipeline for dependency injection during testing."""

    def __init__(self, responses=None, fail_queries=None) -> None:
        self.responses = responses or {}
        self.fail_queries = fail_queries or set()

    def run(self, query: str, top_k=None) -> RAGResponse:
        if query in self.fail_queries:
            raise RuntimeError("Pipeline execution error")
        return self.responses.get(
            query,
            RAGResponse(question=query, answer="Default fallback", sources=[])
        )


def test_rag_evaluator_all_succeed():
    """Verify evaluation when all examples succeed successfully."""
    fake_response = RAGResponse(
        question="What is RAG?",
        answer="Retrieval-Augmented Generation",
        sources=[
            RetrievedSource(content="Content A", source="docA", page_number=1, score=0.95),
            RetrievedSource(content="Content B", source="docB", page_number=1, score=0.80),
        ]
    )
    pipeline = FakePipeline(responses={"What is RAG?": fake_response})
    evaluator = RAGEvaluator(pipeline=pipeline, k=2)

    examples = [
        EvaluationExample(
            query="What is RAG?",
            relevant_document_ids={"docA"},
            expected_answer="RAG combines retrieval and generation."
        )
    ]

    report = evaluator.evaluate(examples)

    assert isinstance(report, EvaluationReport)
    assert report.total_examples == 1
    assert report.successful_examples == 1
    assert report.failed_examples == 0
    assert report.mean_hit_rate_at_k == 1.0
    assert pytest.approx(report.mean_precision_at_k, 0.001) == 0.5
    assert pytest.approx(report.mean_recall_at_k, 0.001) == 1.0
    assert pytest.approx(report.mean_mrr, 0.001) == 1.0
    assert len(report.results) == 1
    assert report.results[0].success is True
    assert report.results[0].generated_answer == "Retrieval-Augmented Generation"


def test_rag_evaluator_one_failure():
    """Verify evaluation handles individual query failures without losing them."""
    success_response = RAGResponse(
        question="Valid Query",
        answer="Valid Answer",
        sources=[
            RetrievedSource(content="Text", source="doc1", page_number=1, score=0.9)
        ]
    )
    pipeline = FakePipeline(
        responses={"Valid Query": success_response},
        fail_queries={"Failing Query"}
    )
    evaluator = RAGEvaluator(pipeline=pipeline, k=2)

    examples = [
        EvaluationExample(query="Valid Query", relevant_document_ids={"doc1"}, expected_answer="Exp 1"),
        EvaluationExample(query="Failing Query", relevant_document_ids={"doc2"}, expected_answer="Exp 2"),
    ]

    report = evaluator.evaluate(examples)

    assert report.total_examples == 2
    assert report.successful_examples == 1
    assert report.failed_examples == 1
    assert report.results[0].success is True
    assert report.results[1].success is False
    assert report.results[1].error_message == "Pipeline execution error"
    assert report.mean_hit_rate_at_k == 1.0


def test_rag_evaluator_empty_dataset():
    """Verify safe handling of an empty evaluation dataset."""
    pipeline = FakePipeline()
    evaluator = RAGEvaluator(pipeline=pipeline, k=3)

    report = evaluator.evaluate([])

    assert report.total_examples == 0
    assert report.successful_examples == 0
    assert report.failed_examples == 0
    assert report.mean_precision_at_k == 0.0
    assert report.results == []


@pytest.mark.parametrize("invalid_k", [0, -1, "3", 2.5, None])
def test_rag_evaluator_invalid_k_raises_error(invalid_k):
    """Verify that invalid k raises ValueError or TypeError."""
    with pytest.raises((ValueError, TypeError)):
        RAGEvaluator(pipeline=FakePipeline(), k=invalid_k)  # type: ignore


def test_rag_evaluator_invalid_pipeline_raises_error():
    """Verify that passing None as pipeline raises ValueError."""
    with pytest.raises(ValueError):
        RAGEvaluator(pipeline=None, k=2)  # type: ignore