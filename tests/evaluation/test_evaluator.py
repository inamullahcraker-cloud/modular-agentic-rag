import pytest

from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.models import RetrievalMetricsSummary, RetrievalTestCase


def test_retrieval_evaluator_success():
    """Verify that RetrievalEvaluator correctly computes aggregate mean metrics."""
    evaluator = RetrievalEvaluator(k=2)
    test_cases = [
        RetrievalTestCase(
            query="What is RAG?",
            relevant_doc_ids={"docA"},
            retrieved_doc_ids=["docA", "docB"],
        ),
        RetrievalTestCase(
            query="How do vector stores work?",
            relevant_doc_ids={"docC"},
            retrieved_doc_ids=["docB", "docC"],
        ),
    ]

    summary = evaluator.evaluate(test_cases)

    assert isinstance(summary, RetrievalMetricsSummary)
    assert summary.total_test_cases == 2
    assert summary.k == 2
    assert pytest.approx(summary.mean_hit_rate_at_k, 0.001) == 1.0
    assert pytest.approx(summary.mean_precision_at_k, 0.001) == 0.5  # (1/2 + 1/2) / 2
    assert pytest.approx(summary.mean_recall_at_k, 0.001) == 1.0     # (1/1 + 1/1) / 2
    assert pytest.approx(summary.mean_mrr, 0.001) == 0.75            # (1.0 + 0.5) / 2


def test_retrieval_evaluator_empty_dataset():
    """Verify safe handling of an empty test dataset."""
    evaluator = RetrievalEvaluator(k=3)
    summary = evaluator.evaluate([])

    assert summary.total_test_cases == 0
    assert summary.mean_precision_at_k == 0.0
    assert summary.mean_recall_at_k == 0.0
    assert summary.mean_hit_rate_at_k == 0.0
    assert summary.mean_mrr == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1, "2", 1.5, None])
def test_retrieval_evaluator_invalid_k_raises_error(invalid_k):
    """Verify that initializing RetrievalEvaluator with invalid k raises ValueError."""
    with pytest.raises((ValueError, TypeError)):
        RetrievalEvaluator(k=invalid_k)  # type: ignore