from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def resolve_device(requested_device: str) -> str:
    """Resolve an automatic or explicit embedding device."""

    normalized_device = requested_device.strip().lower()

    if normalized_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")

    if normalized_device not in {"cpu", "cuda"}:
        raise ValueError("embedding device must be one of: auto, cpu, cuda")

    return normalized_device


class DenseEmbedder:
    """Generate normalized dense text embeddings."""

    def __init__(
        self,
        *,
        model_name: str,
        requested_device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(requested_device)

        print(f"Loading embedding model: {self.model_name}\nEmbedding device: {self.device}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        embedding_dimension = self.model.get_sentence_embedding_dimension()

        if embedding_dimension is None:
            raise RuntimeError("The embedding model did not declare an embedding dimension.")

        self.embedding_dimension = embedding_dimension

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate normalized document vectors."""

        if not texts:
            return np.empty(
                shape=(0, self.embedding_dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> list[float]:
        """Generate one normalized query vector."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The search query cannot be empty.")

        embedding = self.model.encode(
            normalized_query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        ).tolist()
