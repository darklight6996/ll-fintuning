"""
core/services/rag_service.py

Singleton service that wraps the RAGEngine.

Delegates to the existing get_global_rag_engine() singleton so that
the FAISS index and embedding/rerank models are only loaded once.
Consumers call get_rag_service() instead of importing RAGEngine directly,
making it straightforward to swap retrieval backends in future phases.

Usage:
    from core.services.rag_service import get_rag_service
    svc = get_rag_service()
    if svc:
        ranked = svc.retrieve("privilege escalation via SUID binary")
        context = svc.build_context(ranked)
"""

from typing import Optional, List, Tuple, Dict, Any

_INSTANCE: Optional["RAGService"] = None
_LOAD_ATTEMPTED: bool = False


class RAGService:
    """
    Facade over RAGEngine.

    Responsibilities
    ----------------
    * Provide retrieve() and build_context() via a stable interface.
    * Shield callers from RAGEngine import / FAISS load details.
    * Gracefully degrade (return empty results) if index is missing.
    """

    def __init__(self, top_k: int = 5, final_k: int = 2, verbose: bool = False):
        self._engine = None
        self._load(top_k=top_k, final_k=final_k, verbose=verbose)

    def _load(self, top_k: int, final_k: int, verbose: bool) -> None:
        try:
            from core.rag_engine import get_global_rag_engine
            engine = get_global_rag_engine(top_k=top_k, final_k=final_k, verbose=verbose)
            if engine is not None:
                self._engine = engine
                if verbose:
                    print(f"[+] RAGService ready. Index chunks: {len(engine.metadata or [])}")
            else:
                if verbose:
                    print("[!] RAGService: FAISS index not found — retrieval disabled.")
        except Exception as exc:
            print(f"[!] RAGService load failed ({type(exc).__name__}: {exc})")
            self._engine = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the FAISS index loaded successfully."""
        return self._engine is not None

    def retrieve(self, query: str, min_score: float = -2.5) -> List[Tuple[Dict, float]]:
        """
        Retrieve and rerank top documents for a query.

        Parameters
        ----------
        query : str
            The operator's search query.
        min_score : float
            Minimum reranker score to include.

        Returns
        -------
        list[tuple[dict, float]]
            Ranked (doc_metadata, similarity_score) pairs. Empty if unavailable.
        """
        if not self.is_available:
            return []
        try:
            return self._engine.retrieve(query, min_score=min_score)
        except Exception:
            return []

    def build_context(
        self,
        ranked_docs: List[Tuple[Dict, float]],
        min_sim: float = 0.35,
    ) -> str:
        """
        Convert reranked docs into a formatted context string for prompting.

        Parameters
        ----------
        ranked_docs : list[tuple[dict, float]]
            Output from retrieve().
        min_sim : float
            Minimum similarity to include in context block.

        Returns
        -------
        str
            Formatted RAG context block, or empty string if nothing passes threshold.
        """
        if not self.is_available or not ranked_docs:
            return ""
        try:
            return self._engine.build_context(ranked_docs, min_sim=min_sim)
        except Exception:
            return ""

    @property
    def engine(self):
        """Direct access to the underlying RAGEngine (for advanced use)."""
        return self._engine


def get_rag_service(
    top_k: int = 5,
    final_k: int = 2,
    verbose: bool = False,
) -> Optional[RAGService]:
    """
    Return the process-wide RAGService singleton.
    Returns None if the FAISS index is missing or load failed.

    Parameters
    ----------
    top_k : int
        Candidates to retrieve from FAISS before reranking.
    final_k : int
        Number of results to return after reranking.
    verbose : bool
        Print load status on first initialization.
    """
    global _INSTANCE, _LOAD_ATTEMPTED
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        _INSTANCE = RAGService(top_k=top_k, final_k=final_k, verbose=verbose)
        if not _INSTANCE.is_available:
            _INSTANCE = None
    return _INSTANCE
