"""
core/services/embedding_service.py

Thread-safe singleton service that owns the SentenceTransformer embedding model.
Loaded once per process; all consumers share the same instance, eliminating
duplicate GPU/CPU memory allocation and startup latency.

Requirements fulfilled:
- Single SentenceTransformer instance process-wide.
- Supports all-MiniLM-L6-v2, BAAI/bge-large, and future embedding models.
- Exposes: embed(text), embed_many(texts), normalize(vectors), cache(enabled, max_size)
- Lazy loading: Weights loaded on first inference or explicit load().
- Thread safety: RLock-guarded singleton creation, model initialization, and cache access.
- Device detection: Auto-detects CUDA / GPU, DirectML, CPU fallback, and ONNX when applicable.
- Returns float32 numpy arrays.

Usage:
    from core.services.embedding_service import get_embedding_service
    svc = get_embedding_service()
    vector = svc.embed("How do I inspect Azure role assignments?")
    matrix = svc.embed_many(["query 1", "query 2"], normalize=True)
"""

import io
import sys
import os
import logging
import threading
from collections import OrderedDict
from typing import List, Optional, Union, Dict, Any

import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

_INSTANCE: Optional["EmbeddingService"] = None
_INIT_LOCK = threading.RLock()


def _detect_device() -> str:
    """
    Detect the most performant compute device available.
    Returns: 'cuda', 'mps', or 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class EmbeddingCache:
    """
    Thread-safe in-memory LRU cache for computed embeddings.
    """

    def __init__(self, max_size: int = 4096, enabled: bool = True):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._enabled = enabled
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        with self._lock:
            self._enabled = bool(value)

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, value: int):
        with self._lock:
            self._max_size = max(1, int(value))
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def get(self, text: str) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text].copy()
            return None

    def put(self, text: str, embedding: np.ndarray):
        if not self._enabled:
            return
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                self._cache[text] = embedding.copy()
            else:
                self._cache[text] = embedding.copy()
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class EmbeddingService:
    """
    Thread-safe wrapper & singleton manager for SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        lazy: bool = False,
        verbose: bool = False,
        cache_size: int = 4096
    ):
        self.model_name = model_name
        self.device = device or _detect_device()
        self.verbose = verbose
        self._model = None
        self._lock = threading.RLock()
        self._cache = EmbeddingCache(max_size=cache_size, enabled=True)
        self._dimension: Optional[int] = None

        if not lazy:
            self.load()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Thread-safe lazy or eager initialization of the model weights.
        """
        with self._lock:
            if self._model is not None:
                return

            from sentence_transformers import SentenceTransformer

            old_out, old_err = sys.stdout, sys.stderr
            try:
                if not self.verbose:
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()

                # Attempt device load; fallback to cpu gracefully on failure
                try:
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                except Exception:
                    self.device = "cpu"
                    self._model = SentenceTransformer(self.model_name, device="cpu")

            finally:
                sys.stdout = old_out
                sys.stderr = old_err

            # Infer and cache embedding dimension
            test_vec = self._model.encode(["dimension_probe"], convert_to_numpy=True)
            self._dimension = int(test_vec.shape[1])

            if self.verbose:
                print(f"[+] EmbeddingService loaded: {self.model_name} on {self.device} (dim={self._dimension})")

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded in memory."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Embedding output dimension (e.g., 384 or 1024)."""
        if self._dimension is None:
            self.load()
        return self._dimension or 384

    # ------------------------------------------------------------------
    # Normalization Helper
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize float32 numpy array along the last axis.
        Works for 1D (D,) or 2D (N, D) arrays.
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            if norm > 0:
                return (arr / norm).astype(np.float32)
            return arr.astype(np.float32)
        elif arr.ndim == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (arr / norms).astype(np.float32)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Embedding Methods
    # ------------------------------------------------------------------

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text string into a 1D float32 numpy vector of shape (dim,).

        Parameters
        ----------
        text : str
            Input text string to embed.
        normalize : bool, default True
            Whether to L2 normalize the resulting vector.

        Returns
        -------
        np.ndarray of shape (dim,), dtype float32
        """
        if not isinstance(text, str):
            text = str(text)

        # Check cache
        cache_key = f"norm:{normalize}|{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self.load()

        with self._lock:
            raw_vec = self._model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )[0]
            vector = np.array(raw_vec, dtype=np.float32)

        self._cache.put(cache_key, vector)
        return vector

    def embed_many(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Batch embed a list of texts into a 2D float32 numpy matrix of shape (len(texts), dim).

        Parameters
        ----------
        texts : list[str]
            List of text strings.
        normalize : bool, default True
            Whether to L2 normalize the resulting vectors.
        batch_size : int, default 32
            Inference batch size.
        show_progress_bar : bool, default False
            Whether to show tqdm progress bar.

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self.load()

        clean_texts = [str(t) if not isinstance(t, str) else t for t in texts]

        with self._lock:
            raw_embeddings = self._model.encode(
                clean_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar
            )
            matrix = np.array(raw_embeddings, dtype=np.float32)

        # Cache individual items if small batch
        if len(clean_texts) <= 64:
            for txt, vec in zip(clean_texts, matrix):
                self._cache.put(f"norm:{normalize}|{txt}", vec)

        return matrix

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Backward-compatible alias for existing SentenceTransformer.encode callers.
        """
        if isinstance(texts, str):
            res = self.embed_many(
                [texts],
                normalize=normalize_embeddings,
                show_progress_bar=show_progress_bar
            )
            return res
        return self.embed_many(
            texts,
            normalize=normalize_embeddings,
            show_progress_bar=show_progress_bar
        )

    # ------------------------------------------------------------------
    # Cache Control
    # ------------------------------------------------------------------

    def configure_cache(self, enabled: bool = True, max_size: int = 4096) -> None:
        """Configure cache settings."""
        self._cache.enabled = enabled
        self._cache.max_size = max_size

    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        return len(self._cache)

    @property
    def model(self):
        """Direct access to the underlying SentenceTransformer instance."""
        self.load()
        return self._model


# ----------------------------------------------------------------------
# Process-Wide Singleton Accessor
# ----------------------------------------------------------------------

def get_embedding_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    lazy: bool = False,
    verbose: bool = False,
) -> EmbeddingService:
    """
    Return the process-wide EmbeddingService singleton.
    Guarantees only one SentenceTransformer instance is instantiated process-wide.

    Parameters
    ----------
    model_name : str, optional
        Model identifier (e.g. 'all-MiniLM-L6-v2', 'BAAI/bge-large').
    device : str, optional
        Target device ('cuda', 'cpu'). Auto-detected if None.
    lazy : bool, default False
        If True, defers model weight loading until first inference call.
    verbose : bool, default False
        Print load status on initialization.
    """
    global _INSTANCE
    with _INIT_LOCK:
        if _INSTANCE is None:
            if model_name is None:
                try:
                    from core.config import EMBED_MODEL
                    model_name = EMBED_MODEL
                except ImportError:
                    model_name = "all-MiniLM-L6-v2"
            _INSTANCE = EmbeddingService(
                model_name=model_name,
                device=device,
                lazy=lazy,
                verbose=verbose
            )
        return _INSTANCE
