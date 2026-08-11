from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    """Fraction of relevant chunks found in the top k results."""

    if not relevant_ids:
        return None

    retrieved_top_k = set(retrieved_ids[:k])

    return len(retrieved_top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""

    retrieved_top_k = retrieved_ids[:k]

    if not retrieved_top_k:
        return 0.0

    hits = sum(1 for chunk_id in retrieved_top_k if chunk_id in relevant_ids)

    return hits / len(retrieved_top_k)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant result, or 0 if none was found."""

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def _dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 1) for index, rel in enumerate(relevances[:k], start=1) if rel)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at k, using binary relevance."""

    relevances = [1 if chunk_id in relevant_ids else 0 for chunk_id in retrieved_ids[:k]]
    dcg = _dcg_at_k(relevances, k)

    ideal_relevances = [1] * min(len(relevant_ids), k)
    idcg = _dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0