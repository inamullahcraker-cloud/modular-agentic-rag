"""Evaluation runner module for aggregating retrieval performance across a test dataset."""

from typing import Sequence

from src.evaluation.models import RetrievalMetricsSummary, RetrievalTestCase
from src.evaluation.retrieval_metrics import (
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


class RetrievalEvaluator:
    """Orchestrates retrieval evaluation across multiple test cases and computes aggregate metrics."""

    def __init__(self, k: int = 3) -> None:
        """Initializes the evaluator with a target cutoff rank K.

        Args:
            k: Positive integer cutoff for top-K evaluation.

        Raises:
            ValueError: If k is not a positive integer.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer greater than 0, got {k!r}")
        self.k = k

    def evaluate(self, test_cases: Sequence[RetrievalTestCase]) -> RetrievalMetricsSummary:
        """Evaluates a sequence of retrieval test cases and returns summary averages.

        Args:
            test_cases: Sequence of RetrievalTestCase instances.

        Returns:
            RetrievalMetricsSummary containing mean metrics across all cases.
        """
        if not test_cases:
            return RetrievalMetricsSummary(
                total_test_cases=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_hit_rate_at_k=0.0,
                mean_mrr=0.0,
                k=self.k,
            )

        precisions = []
        recalls = []
        hit_rates = []
        mrrs = []

        for case in test_cases:
            precisions.append(calculate_precision_at_k(case.retrieved_doc_ids, case.relevant_doc_ids, self.k))
            recalls.append(calculate_recall_at_k(case.retrieved_doc_ids, case.relevant_doc_ids, self.k))
            hit_rates.append(calculate_hit_rate_at_k(case.retrieved_doc_ids, case.relevant_doc_ids, self.k))
            mrrs.append(calculate_mrr(case.retrieved_doc_ids, case.relevant_doc_ids, k=self.k))

        total = len(test_cases)
        return RetrievalMetricsSummary(
            total_test_cases=total,
            mean_precision_at_k=sum(precisions) / total,
            mean_recall_at_k=sum(recalls) / total,
            mean_hit_rate_at_k=sum(hit_rates) / total,
            mean_mrr=sum(mrrs) / total,
            k=self.k,
        )