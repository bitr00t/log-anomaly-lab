"""Sentence-transformers wrapper.

Same isolation pattern as in rag-eval-lab: nothing outside this module
imports sentence-transformers, so swapping the embedding backend is a
one-file change and unit tests never pull in torch.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts -> (n, dim) float32 array, L2-normalised.

        Normalisation matters twice here: it makes cosine similarity in
        Qdrant correct, and it puts Euclidean distances (used by the
        sklearn detectors) on a bounded, comparable scale.
        """
        return self._model.encode(
            texts, batch_size=128, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)
