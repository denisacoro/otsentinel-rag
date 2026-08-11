from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-scores retrieved passages against the query using a cross-encoder."""

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        print(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Score (chunk_id, text) candidates against the query, sorted best first."""

        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)

        scored = [
            (chunk_id, float(score)) for (chunk_id, _), score in zip(candidates, scores, strict=True)
        ]

        scored.sort(key=lambda item: item[1], reverse=True)

        return scored