from __future__ import annotations

from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from qdrant_client import models


class SparseEmbedder:
    """Generate BM25-style sparse vectors, meant to be used with Qdrant's IDF modifier."""

    def __init__(self, *, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        print(f"Loading sparse embedding model: {self.model_name}")
        self.model = SparseTextEmbedding(model_name=self.model_name)

    def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Generate document-side sparse vectors (term-frequency weighted)."""

        embeddings = list(self.model.embed(list(texts)))

        return [
            models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
            for embedding in embeddings
        ]

    def encode_query(self, query: str) -> models.SparseVector:
        """Generate a query-side sparse vector (binary term presence)."""

        embedding = next(iter(self.model.query_embed(query)))

        return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())