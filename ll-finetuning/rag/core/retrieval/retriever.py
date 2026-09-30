"""
core/retrieval/retriever.py

Modular document retriever adhering to Dependency Injection and Single Responsibility principles.
Coordinates query embedding, vector store retrieval, and chunk filtering.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from core.retrieval.vector_store import BaseVectorStore
from core.services.embedding_service import EmbeddingService, get_embedding_service


class ChunkFilter:
    """
    Filters retrieved vector store candidate chunks based on similarity thresholds
    and metadata constraints.
    """

    def __init__(self, default_min_sim: float = 0.25):
        self.default_min_sim = default_min_sim

    def filter(
        self,
        candidates: List[Tuple[Dict[str, Any], float]],
        min_sim: Optional[float] = None,
        required_keys: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        threshold = self.default_min_sim if min_sim is None else min_sim
        filtered = []

        for doc, score in candidates:
            sim = doc.get("_score", score)
            if sim < threshold:
                continue

            if required_keys:
                if not all(k in doc for k in required_keys):
                    continue

            filtered.append((doc, float(sim)))

        return filtered


class Retriever:
    """
    Decoupled retriever that combines an EmbeddingService, a BaseVectorStore,
    and a ChunkFilter.

    No model instantiation occurs inside this class — all components are injected.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        chunk_filter: Optional[ChunkFilter] = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        self.chunk_filter = chunk_filter or ChunkFilter()

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query into float32 normalized vector."""
        return self.embedding_service.embed_many([query])

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_sim: float = 0.25
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Embed query, query the vector store, and filter candidates.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of (document_metadata, similarity_score).
        """
        if not self.vector_store.is_ready():
            return []

        query_vec = self.embed_query(query)
        raw_candidates = self.vector_store.search(query_vec, top_k=top_k)

        filtered = self.chunk_filter.filter(raw_candidates, min_sim=min_sim)
        return filtered
