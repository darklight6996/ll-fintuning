"""
core/retrieval/reranker.py

Cross-encoder reranker module adhering to Dependency Injection.
Reranks initial vector retrieval candidates using neural sequence cross-attention.
"""

from typing import List, Tuple, Dict, Any, Optional
from core.services.rerank_service import RerankService, get_rerank_service


class Reranker:
    """
    Reranks document candidates against a user query using a CrossEncoder.
    Uses Dependency Injection — models are never instantiated directly.
    """

    def __init__(
        self,
        rerank_service: Optional[RerankService] = None
    ):
        self.rerank_service = rerank_service or get_rerank_service()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        min_score: float = -2.5,
        final_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Compute cross-attention relevance scores for query-document pairs.

        Parameters
        ----------
        query : str
            User query string.
        candidates : List[Tuple[Dict[str, Any], float]]
            List of (metadata_dict, vector_similarity) from retriever.
        min_score : float
            Logit threshold; candidates below this are dropped.
        final_k : int
            Maximum number of top results to return.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            Reranked documents sorted descending by cross-encoder score.
        """
        if not candidates or self.rerank_service is None or not self.rerank_service.is_available:
            # Fallback: sort by initial vector similarity
            sorted_c = sorted(candidates, key=lambda x: x[0].get("_score", x[1]), reverse=True)
            return sorted_c[:final_k]

        pairs = [[query, doc.get("text", "")] for doc, _ in candidates]
        scores = self.rerank_service.predict(pairs)

        ranked = []
        for (doc, sim), score in zip(candidates, scores):
            score_f = float(score)
            doc["_rerank_score"] = score_f
            if score_f >= min_score:
                ranked.append((doc, float(sim)))

        ranked.sort(key=lambda x: x[0].get("_rerank_score", x[1]), reverse=True)
        return ranked[:final_k]
