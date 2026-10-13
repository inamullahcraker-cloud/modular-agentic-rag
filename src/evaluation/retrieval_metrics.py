"""Retrieval evaluation metrics module for RAG pipelines.

Provides deterministic, ID-based metrics: Precision@K, Recall@K, Hit Rate@K, and MRR.
"""

from typing import Optional, Sequence, Set


def _validate_k(k: int) -> None:
    """Validates that k is a positive integer greater than zero.

    Raises:
        ValueError: If k is not a positive integer or is a boolean.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer greater than 0, got {k!r}")


def calculate_precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> float:
    """Calculates Precision@K.

    Definition:
        Relevant retrieved documents in the top K positions divided by K.

    Policy for missing/empty ground truth (relevant_ids):
        If `relevant_ids` is empty, returns 0.0 as there is no ground truth
        to evaluate against.
    """
    _validate_k(k)
    if not relevant_ids or not retrieved_ids:
        return 0.0

    top_k_retrieved = list(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)
    intersection_count = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)

    return intersection_count / k


def calculate_recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> float:
    """Calculates Recall@K.

    Definition:
        Relevant retrieved documents in the top K positions divided by the total
        number of known relevant documents.

    Policy for missing/empty ground truth (relevant_ids):
        If `relevant_ids` is empty, returns 0.0 to avoid division-by-zero errors
        and signify the absence of verifiable ground truth.
    """
    _validate_k(k)
    relevant_set = set(relevant_ids)
    if not relevant_set or not retrieved_ids:
        return 0.0

    top_k_retrieved = list(retrieved_ids)[:k]
    intersection_count = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)

    return intersection_count / len(relevant_set)


def calculate_hit_rate_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> float:
    """Calculates Hit Rate@K.

    Definition:
        1.0 if at least one relevant document is retrieved within the top K positions;
        otherwise 0.0.

    Policy for missing/empty ground truth (relevant_ids):
        If `relevant_ids` is empty, returns 0.0 as no ground truth exists.
    """
    _validate_k(k)
    if not relevant_ids or not retrieved_ids:
        return 0.0

    top_k_retrieved = list(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)

    for doc_id in top_k_retrieved:
        if doc_id in relevant_set:
            return 1.0

    return 0.0


def calculate_mrr(
    retrieved_ids: Sequence[str], relevant_ids: Set[str], k: Optional[int] = None
) -> float:
    """Calculates Mean Reciprocal Rank (MRR).

    Definition:
        Reciprocal rank ($1 / \text{rank}$) of the first relevant retrieved document
        within the top K results (if K is specified) or across all retrieved results.

    Policy for missing/empty ground truth (relevant_ids):
        If `relevant_ids` is empty, returns 0.0 as no ground truth exists.
    """
    if k is not None:
        _validate_k(k)

    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    eval_list = list(retrieved_ids)[:k] if k is not None else list(retrieved_ids)

    for rank, doc_id in enumerate(eval_list, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0