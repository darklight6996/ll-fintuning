"""
core/services/rerank_service.py

Singleton service that owns the CrossEncoder reranker model.
Loaded once per process; all consumers share the same instance.

Usage:
    from core.services.rerank_service import get_rerank_service
    svc = get_rerank_service()
    scores = svc.predict([["query", "doc text"], ...])
"""

import io
import sys
import logging
from typing import List, Optional

import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_INSTANCE: Optional["RerankService"] = None


class RerankService:
    """
    Thin wrapper around CrossEncoder.

    Responsibilities
    ----------------
    * Load the reranker model once.
    * Expose predict() as a clean interface.
    """

    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        self._model = None
        self._load(verbose=verbose)

    def _load(self, verbose: bool = False) -> None:
        from sentence_transformers import CrossEncoder

        old_out, old_err = sys.stdout, sys.stderr
        try:
            if not verbose:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
            self._model = CrossEncoder(self.model_name)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        if verbose:
            print(f"[+] RerankService ready: {self.model_name}")

    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """
        Score query-document pairs.

        Parameters
        ----------
        pairs : list[list[str]]
            Each element is [query, document_text].

        Returns
        -------
        np.ndarray of float scores (higher = more relevant).
        """
        return self._model.predict(pairs)

    @property
    def model(self):
        """Direct access to the underlying CrossEncoder (for advanced use)."""
        return self._model


def get_rerank_service(
    model_name: Optional[str] = None,
    verbose: bool = False,
) -> RerankService:
    """
    Return the process-wide RerankService singleton.
    Creates it on first call; subsequent calls return the cached instance.

    Parameters
    ----------
    model_name : str, optional
        Override the model name from config. Only honoured on the first call.
    verbose : bool
        Print load status on first initialization.
    """
    global _INSTANCE
    if _INSTANCE is None:
        if model_name is None:
            from core.config import RERANK_MODEL
            model_name = RERANK_MODEL
        _INSTANCE = RerankService(model_name=model_name, verbose=verbose)
    return _INSTANCE
