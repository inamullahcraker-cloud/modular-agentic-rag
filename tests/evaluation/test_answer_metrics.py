"""Unit tests for answer quality and groundedness evaluation metrics."""

import pytest

from src.evaluation.answer_metrics import (
    calculate_answer_relevance,
    calculate_context_coverage,
    calculate_groundedness,
)


def test_answer_relevance_strong_and_weak():
    """Verify answer relevance score handles strong vs weak lexical overlap."""
    question = "What is a vector database?"
    strong_answer = "A vector database stores and indexes vector embeddings."
    weak_answer = "Databases are useful for storing information."

    strong_score = calculate_answer_relevance(question, strong_answer)
    weak_score = calculate_answer_relevance(question, weak_answer)

    assert strong_score > weak_score
    assert 0.0 <= weak_score <= strong_score <= 1.0


def test_context_coverage_cases():
    """Verify context coverage correctly calculates reference token presence."""
    reference = "Vector databases enable fast semantic search."
    contexts = [
        "Vector databases store embeddings.",
        "They enable fast semantic search capabilities.",
    ]

    score = calculate_context_coverage(reference, contexts)
    assert 0.0 < score <= 1.0


def test_groundedness_cases():
    """Verify groundedness proxy calculates generated answer support in context."""
    answer = "Vector databases store embeddings for retrieval."
    contexts = ["Vector databases store high-dimensional embeddings for fast retrieval."]

    score = calculate_groundedness(answer, contexts)
    assert pytest.approx(score, 0.001) == 1.0


def test_empty_inputs_handled():
    """Verify all metrics safely return 0.0 when given empty strings or empty context lists."""
    assert calculate_answer_relevance("", "Some answer") == 0.0
    assert calculate_answer_relevance("Some question?", "") == 0.0

    assert calculate_context_coverage("", ["Context"]) == 0.0
    assert calculate_context_coverage("Reference", []) == 0.0

    assert calculate_groundedness("", ["Context"]) == 0.0
    assert calculate_groundedness("Answer", []) == 0.0


@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, ["not a string"]]
)
def test_invalid_input_types_raise_error(invalid_input):
    """Verify that non-string inputs raise TypeError."""
    with pytest.raises(TypeError):
        calculate_answer_relevance(invalid_input, "answer")  # type: ignore

    with pytest.raises(TypeError):
        calculate_answer_relevance("question", invalid_input)  # type: ignore

    with pytest.raises(TypeError):
        calculate_context_coverage(invalid_input, ["ctx"])  # type: ignore

    with pytest.raises(TypeError):
        calculate_context_coverage("ref", None)  # type: ignore

    with pytest.raises(TypeError):
        calculate_groundedness(invalid_input, ["ctx"])  # type: ignore