# ==========================================================
# core/rag_engine.py
# Retrieval + Reranking Engine for RAG
# Refactored for Phase 4: Modular architecture with Dependency Injection,
# pluggable vector store adapters (FAISS, HNSW, Qdrant, Chroma),
# and separated Retriever, Reranker, and ContextBuilder.
# ==========================================================

# ==============================
# IMPORTS
# ==============================

import os
import sys
import logging
import warnings
from typing import Optional, List, Tuple, Dict, Any

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np

from core.config import (
    INDEX_PATH,
    META_PATH,
    TOP_K,
    FINAL_K,
    EMBED_MODEL,
    RERANK_MODEL,
    VECTOR_STORE_BACKEND
)

# Services layer — singletons shared across the process
from core.services.embedding_service import get_embedding_service, EmbeddingService
from core.services.rerank_service import get_rerank_service, RerankService

# Retrieval subsystem — decoupled adapters and components
from core.retrieval.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    create_vector_store
)
from core.retrieval.retriever import Retriever, ChunkFilter
from core.retrieval.reranker import Reranker
from core.retrieval.context_builder import RAGContextBuilder


_GLOBAL_RAG_ENGINE: Optional["RAGEngine"] = None


# ==============================
# RAG ENGINE CLASS
# ==============================

class RAGEngine:
    """
    RAGEngine coordinates modular RAG operations:
    1. Query embedding (via shared EmbeddingService)
    2. Vector similarity retrieval (via BaseVectorStore adapter)
    3. Neural cross-encoder reranking (via Reranker)
    4. Formatted context generation (via RAGContextBuilder)

    Supports full Dependency Injection while preserving 100% backward
    compatibility for existing callers.
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH,
        top_k: int = TOP_K,
        final_k: int = FINAL_K,
        embed_model_name: str = EMBED_MODEL,
        rerank_model_name: str = RERANK_MODEL,
        vector_store: Optional[BaseVectorStore] = None,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        context_builder: Optional[RAGContextBuilder] = None,
        backend: Optional[str] = None,
        verbose: bool = False
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.top_k = top_k
        self.final_k = final_k
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name
        self.verbose = verbose
        self.backend = backend or VECTOR_STORE_BACKEND

        # Dependency Injection / Component Initialization
        self.embedding_service = get_embedding_service(model_name=self.embed_model_name, verbose=verbose)
        self.rerank_service = get_rerank_service(model_name=self.rerank_model_name, verbose=verbose)

        # Backward compatibility properties
        self.embed_model = self.embedding_service.model if self.embedding_service else None
        self.rerank_model = self.rerank_service.model if self.rerank_service else None

        # 1. Vector Store Adapter
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = create_vector_store(
                backend=self.backend,
                index_path=self.index_path,
                meta_path=self.meta_path
            )

        # Backward compatibility aliases for direct FAISS index / metadata access
        if isinstance(self.vector_store, FAISSVectorStore):
            self.index = self.vector_store.index
            self.metadata = self.vector_store.metadata
        else:
            self.index = getattr(self.vector_store, "index", None)
            self.metadata = getattr(self.vector_store, "metadata", [])

        # 2. Decoupled Retriever
        self.retriever = retriever or Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            chunk_filter=ChunkFilter(default_min_sim=0.25)
        )

        # 3. Decoupled Reranker
        self.reranker_component = reranker or Reranker(
            rerank_service=self.rerank_service
        )
        # Expose reranker model directly for legacy callers
        self.reranker = self.rerank_model

        # 4. Decoupled Context Builder
        self.context_builder = context_builder or RAGContextBuilder(default_min_sim=0.35)

        if verbose:
            print(f"[+] RAG Neural Engine ready (backend={self.backend}).")

    # ==========================================================
    # RETRIEVAL & RERANKING
    # ==========================================================

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert a user query into a float32 embedding."""
        return self.retriever.embed_query(query)

    def search_similarity(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute vector similarity search returning scores normalized as 1 / (1 + distance),
        rather than raw L2 Euclidean distance.
        """
        k = top_k or self.top_k
        if not self.vector_store or not self.vector_store.is_ready():
            return []
        query_vec = self._embed_query(query)
        results = self.vector_store.search(query_vec, top_k=k)
        normalized = []
        for meta, score in results:
            sim = meta.get("_score", score)
            if "_faiss_dist" in meta:
                sim = 1.0 / (1.0 + float(meta["_faiss_dist"]))
                meta["_score"] = float(sim)
            normalized.append((meta, float(sim)))
        return normalized

    def retrieve(self, query: str, min_score: float = -2.5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top documents for a query and rerank them, filtering out negative irrelevant logits.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of tuples: [(doc_metadata_dict, reranker_score), ...]
        """
        # Step 1: Vector retrieval + similarity filtering
        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            min_sim=0.25
        )

        if not candidates:
            return []

        # Step 2: Cross-encoder reranking
        ranked = self.reranker_component.rerank(
            query=query,
            candidates=candidates,
            min_score=min_score,
            final_k=self.final_k
        )

        return ranked

    # ==========================================================
    # CONTEXT BUILDER
    # ==========================================================

    def build_context(self, ranked_docs: List[Tuple[Dict[str, Any], float]], min_sim: float = 0.35) -> str:
        """
        Convert reranked docs into one formatted context string.
        Filters out low-similarity chunks below min_sim (default 0.35).
        """
        return self.context_builder.build(ranked_docs, min_sim=min_sim)

    # ==========================================================
    # OPTIONAL DEBUG INFO
    # ==========================================================

    def info(self) -> dict:
        """Return engine configuration for debugging."""
        return {
            "backend": self.backend,
            "index_path": self.index_path,
            "meta_path": self.meta_path,
            "top_k": self.top_k,
            "final_k": self.final_k,
            "embed_model": self.embed_model_name,
            "rerank_model": self.rerank_model_name,
            "metadata_count": self.vector_store.count() if self.vector_store else 0,
            "is_ready": self.vector_store.is_ready() if self.vector_store else False
        }


def get_global_rag_engine(
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
    verbose: bool = False
) -> Optional[RAGEngine]:
    """
    Module-level singleton getter to ensure model weights and FAISS indices
    are loaded only once in process memory, eliminating reload lag.
    """
    global _GLOBAL_RAG_ENGINE
    if _GLOBAL_RAG_ENGINE is None:
        try:
            from pathlib import Path
            if Path(index_path).exists() and Path(meta_path).exists():
                _GLOBAL_RAG_ENGINE = RAGEngine(
                    index_path=index_path,
                    meta_path=meta_path,
                    top_k=top_k,
                    final_k=final_k,
                    verbose=verbose
                )
            else:
                _GLOBAL_RAG_ENGINE = False
        except Exception as e:
            if verbose:
                print(f"[!] Error initializing global RAGEngine: {e}")
            _GLOBAL_RAG_ENGINE = False

    return _GLOBAL_RAG_ENGINE if _GLOBAL_RAG_ENGINE is not False else None