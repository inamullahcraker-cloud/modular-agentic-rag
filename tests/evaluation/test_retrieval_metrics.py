import pytest

from src.evaluation.retrieval_metrics import (
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


def test_metrics_normal_case():
    """Verify normal calculation matching the prompt example."""
    retrieved = ["B", "C", "A"]
    relevant = {"A", "C"}
    k = 3

    assert pytest.approx(calculate_precision_at_k(retrieved, relevant, k), 0.001) == 2 / 3
    assert pytest.approx(calculate_recall_at_k(retrieved, relevant, k), 0.001) == 1.0
    assert calculate_hit_rate_at_k(retrieved, relevant, k) == 1.0
    assert pytest.approx(calculate_mrr(retrieved, relevant, k), 0.001) == 0.5


def test_metrics_empty_retrieved():
    """Verify metrics return 0.0 when retrieval results are empty."""
    retrieved = []
    relevant = {"A", "C"}
    k = 3

    assert calculate_precision_at_k(retrieved, relevant, k) == 0.0
    assert calculate_recall_at_k(retrieved, relevant, k) == 0.0
    assert calculate_hit_rate_at_k(retrieved, relevant, k) == 0.0
    assert calculate_mrr(retrieved, relevant, k) == 0.0


def test_metrics_empty_relevant():
    """Verify metrics return 0.0 when no relevant documents are known (policy)."""
    retrieved = ["A", "B"]
    relevant = set()
    k = 2

    assert calculate_precision_at_k(retrieved, relevant, k) == 0.0
    assert calculate_recall_at_k(retrieved, relevant, k) == 0.0
    assert calculate_hit_rate_at_k(retrieved, relevant, k) == 0.0
    assert calculate_mrr(retrieved, relevant, k) == 0.0


def test_mrr_accepts_none_k():
    """Verify MRR permits k=None to calculate across all retrieved results."""
    retrieved = ["B", "A", "C"]
    relevant = {"A"}
    assert pytest.approx(calculate_mrr(retrieved, relevant, k=None), 0.001) == 0.5


@pytest.mark.parametrize("invalid_k", [0, -1, -5, "3", 2.5, None])
def test_precision_recall_hit_rate_invalid_k_raises_error(invalid_k):
    """Verify that Precision, Recall, and Hit Rate reject invalid k values including None."""
    retrieved = ["A", "B"]
    relevant = {"A"}

    with pytest.raises((ValueError, TypeError)):
        calculate_precision_at_k(retrieved, relevant, invalid_k)  # type: ignore

    with pytest.raises((ValueError, TypeError)):
        calculate_recall_at_k(retrieved, relevant, invalid_k)  # type: ignore

    with pytest.raises((ValueError, TypeError)):
        calculate_hit_rate_at_k(retrieved, relevant, invalid_k)  # type: ignore


@pytest.mark.parametrize("invalid_k", [0, -1, -5, "3", 2.5])
def test_mrr_invalid_k_raises_error(invalid_k):
    """Verify that MRR rejects malformed k values (excluding None which is valid)."""
    retrieved = ["A", "B"]
    relevant = {"A"}

    with pytest.raises((ValueError, TypeError)):
        calculate_mrr(retrieved, relevant, invalid_k)  # type: ignore