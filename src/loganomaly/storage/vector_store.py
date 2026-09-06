"""Qdrant vector store.

The vector database plays two roles in this project:

1. Index/serving layer: every log line is stored with its payload so
   anomalies can be inspected and their nearest neighbours retrieved
   during triage.
2. Detector backend: the k-NN distance detector computes its anomaly
   score directly from Qdrant search results (mean distance to the k
   nearest neighbours), i.e. the vector DB is part of the model, not
   just storage.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from loganomaly.models import LogRecord

if TYPE_CHECKING:
    from loganomaly.embedding.embedder import Embedder


def _point_id(record_id: str) -> str:
    """Stable UUID derived from the human-readable record id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


class VectorStore:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------------ #
    def recreate_collection(self, name: str, dimension: int) -> None:
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )

    def upsert_records(
        self, collection: str, records: list[LogRecord], vectors: np.ndarray
    ) -> None:
        """Upload records in batches; the label is stored in the payload
        for later inspection but is never read by any detector."""
        batch = 256
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            chunk_vectors = vectors[start : start + batch]
            points = [
                qm.PointStruct(
                    id=_point_id(rec.record_id),
                    vector=vec.tolist(),
                    payload={
                        "record_id": rec.record_id,
                        "timestamp": rec.timestamp,
                        "level": rec.level,
                        "service": rec.service,
                        "message": rec.message,
                        "is_anomaly": rec.is_anomaly,
                        "anomaly_type": rec.anomaly_type,
                    },
                )
                for rec, vec in zip(chunk, chunk_vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)

    # ------------------------------------------------------------------ #
    def knn_mean_distance(self, collection: str, vectors: np.ndarray, k: int = 10) -> np.ndarray:
        """k-NN distance anomaly score, computed via Qdrant.

        For every vector: search its k+1 nearest neighbours (the closest
        hit is the point itself, score ~1.0 cosine similarity), drop self,
        and average (1 - similarity) over the remaining k. Points in dense
        regions get scores near 0; isolated points score high.
        """
        requests = [
            qm.SearchRequest(vector=vec.tolist(), limit=k + 1, with_payload=False)
            for vec in vectors
        ]
        scores = np.zeros(len(vectors), dtype=np.float64)
        batch = 128  # keep individual HTTP requests reasonably sized
        for start in range(0, len(requests), batch):
            responses = self.client.search_batch(
                collection_name=collection, requests=requests[start : start + batch]
            )
            for offset, hits in enumerate(responses):
                # Drop the first hit (the point itself) and convert cosine
                # similarity -> distance.
                neighbour_sims = [h.score for h in hits[1:]]
                scores[start + offset] = float(
                    np.mean([1.0 - s for s in neighbour_sims]) if neighbour_sims else 0.0
                )
        return scores

    def neighbours(self, collection: str, vector: np.ndarray, limit: int = 5) -> list[dict]:
        """Nearest neighbours with payload — used by the triage report to
        show what an anomaly is (dis)similar to."""
        hits = self.client.search(
            collection_name=collection, query_vector=vector.tolist(), limit=limit
        )
        return [{"score": float(h.score), **(h.payload or {})} for h in hits]
