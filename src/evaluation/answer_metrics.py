"""Answer quality and groundedness evaluation metrics module for RAG pipelines.

Provides deterministic, lexical/token-based baselines for:
- Answer relevance (question vs answer overlap)
- Context coverage (reference answer vs retrieved context overlap)
- Groundedness proxy (generated answer vs retrieved context lexical support)
"""

import re
from typing import Sequence


def _tokenize(text: str) -> set[str]:
    """Tokenizes a string into a set of lowercase word tokens.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    words = re.findall(r"\b\w+\b", text.lower())
    return set(words)


def calculate_answer_relevance(question: str, answer: str) -> float:
    """Calculates deterministic lexical relevance between a question and an answer.

    Definition:
        Fraction of unique word tokens in the question that also appear in the answer.

    Limitations:
        - Lexical baseline only: does not capture semantic synonyms (e.g., 'cost' vs 'price')
          or paraphrasing.
        - Measures topical keyword overlap, not logical completeness or correctness.
    """
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)

    if not q_tokens or not a_tokens:
        return 0.0

    intersection = q_tokens.intersection(a_tokens)
    return len(intersection) / len(q_tokens)


def calculate_context_coverage(reference_answer: str, retrieved_contexts: Sequence[str]) -> float:
    """Measures how much of the reference answer is represented in the retrieved context.

    Definition:
        Fraction of unique word tokens in the reference answer found across all
        retrieved context chunks combined.

    Limitations:
        - Lexical check: sensitive to phrasing differences and vocabulary mismatch.
        - Does not account for contextual importance or syntactic nuance.
    """
    if retrieved_contexts is None:
        raise TypeError("retrieved_contexts cannot be None.")

    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(str(ctx) for ctx in retrieved_contexts)
    ctx_tokens = _tokenize(combined_context)

    if not ctx_tokens:
        return 0.0

    intersection = ref_tokens.intersection(ctx_tokens)
    return len(intersection) / len(ref_tokens)


def calculate_groundedness(generated_answer: str, retrieved_contexts: Sequence[str]) -> float:
    """Estimates lexical grounding of the generated answer in the retrieved context.

    Definition:
        Fraction of unique word tokens in the generated answer that appear within
        the retrieved context chunks.

    Limitations:
        - Lexical support proxy ONLY; does NOT prove factual correctness, logical
          entailment, or semantic faithfulness.
        - Can be misled if the context contains matching keywords used in negative,
          irrelevant, or contradictory statements.
    """
    if retrieved_contexts is None:
        raise TypeError("retrieved_contexts cannot be None.")

    gen_tokens = _tokenize(generated_answer)
    if not gen_tokens or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(str(ctx) for ctx in retrieved_contexts)
    ctx_tokens = _tokenize(combined_context)

    if not ctx_tokens:
        return 0.0

    intersection = gen_tokens.intersection(ctx_tokens)
    return len(intersection) / len(gen_tokens)