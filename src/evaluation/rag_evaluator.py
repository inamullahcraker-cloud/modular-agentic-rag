"""End-to-end RAG evaluation runner module."""

import logging
from typing import Any, Sequence

from src.evaluation.models import EvaluationExample, EvaluationReport, SingleEvaluationResult
from src.evaluation.retrieval_metrics import (
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates a RAG pipeline against a collection of evaluation examples."""

    def __init__(self, pipeline: Any, k: int = 3) -> None:
        """Initializes the RAGEvaluator.

        Args:
            pipeline: RAG pipeline instance supporting `.run(query)`.
            k: Positive integer cutoff for top-K retrieval evaluation.

        Raises:
            ValueError: If pipeline is None or k is not a positive integer.
        """
        if pipeline is None:
            raise ValueError("pipeline cannot be None.")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer greater than 0, got {k!r}")

        self.pipeline = pipeline
        self.k = k

    def evaluate(self, examples: Sequence[EvaluationExample]) -> EvaluationReport:
        """Evaluates a sequence of evaluation examples using the RAG pipeline.

        Args:
            examples: Sequence of EvaluationExample instances.

        Returns:
            EvaluationReport containing aggregate metrics and per-example results.
        """
        if not examples:
            logger.info("Evaluation dataset is empty.")
            return EvaluationReport(
                total_examples=0,
                successful_examples=0,
                failed_examples=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_hit_rate_at_k=0.0,
                mean_mrr=0.0,
                k=self.k,
                results=[],
            )

        results: list[SingleEvaluationResult] = []
        successful_count = 0
        failed_count = 0

        precisions = []
        recalls = []
        hit_rates = []
        mrrs = []

        for example in examples:
            try:
                logger.info("Evaluating query: '%s'", example.query)
                response = self.pipeline.run(example.query)

                # Support both response.sources (from schemas.py) and response.retrieved_sources
                sources_list = getattr(response, "sources", getattr(response, "retrieved_sources", []))

                # Extract document IDs/sources from RetrievedSource objects
                retrieved_ids = [
                    getattr(src, "source", getattr(src, "file_name", "Unknown"))
                    for src in sources_list
                ]

                p_k = calculate_precision_at_k(retrieved_ids, example.relevant_document_ids, self.k)
                r_k = calculate_recall_at_k(retrieved_ids, example.relevant_document_ids, self.k)
                hr_k = calculate_hit_rate_at_k(retrieved_ids, example.relevant_document_ids, self.k)
                mrr_val = calculate_mrr(retrieved_ids, example.relevant_document_ids, k=self.k)

                results.append(
                    SingleEvaluationResult(
                        query=example.query,
                        success=True,
                        generated_answer=response.answer,
                        expected_answer=example.expected_answer,
                        retrieved_doc_ids=retrieved_ids,
                        relevant_doc_ids=example.relevant_document_ids,
                        precision_at_k=p_k,
                        recall_at_k=r_k,
                        hit_rate_at_k=hr_k,
                        mrr=mrr_val,
                        error_message=None,
                    )
                )
                successful_count += 1
                precisions.append(p_k)
                recalls.append(r_k)
                hit_rates.append(hr_k)
                mrrs.append(mrr_val)

            except Exception as exc:
                logger.error("Query evaluation failed for '%s': %s", example.query, exc, exc_info=True)
                results.append(
                    SingleEvaluationResult(
                        query=example.query,
                        success=False,
                        generated_answer=None,
                        expected_answer=example.expected_answer,
                        retrieved_doc_ids=[],
                        relevant_doc_ids=example.relevant_document_ids,
                        precision_at_k=0.0,
                        recall_at_k=0.0,
                        hit_rate_at_k=0.0,
                        mrr=0.0,
                        error_message=str(exc),
                    )
                )
                failed_count += 1

        total = len(examples)
        mean_p = sum(precisions) / successful_count if successful_count > 0 else 0.0
        mean_r = sum(recalls) / successful_count if successful_count > 0 else 0.0
        mean_hr = sum(hit_rates) / successful_count if successful_count > 0 else 0.0
        mean_mrr = sum(mrrs) / successful_count if successful_count > 0 else 0.0

        return EvaluationReport(
            total_examples=total,
            successful_examples=successful_count,
            failed_examples=failed_count,
            mean_precision_at_k=mean_p,
            mean_recall_at_k=mean_r,
            mean_hit_rate_at_k=mean_hr,
            mean_mrr=mean_mrr,
            k=self.k,
            results=results,
        )