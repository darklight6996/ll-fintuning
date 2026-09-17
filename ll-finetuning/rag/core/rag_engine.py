# ==========================================================
# core/rag_engine.py
# Retrieval + Reranking Engine for RAG
# ==========================================================

# ==============================
# IMPORTS
# ==============================

import os
import sys
import io
import logging
import warnings
from typing import Optional, List, Tuple, Dict, Any

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import faiss
import pickle
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from core.config import (
    INDEX_PATH,
    META_PATH,
    TOP_K,
    FINAL_K,
    EMBED_MODEL,
    RERANK_MODEL
)

_GLOBAL_RAG_ENGINE: Optional["RAGEngine"] = None


# ==============================
# RAG ENGINE CLASS
# ==============================

class RAGEngine:
    """
    RAGEngine handles:
    1. Loading the embedding model
    2. Loading the reranker
    3. Loading the FAISS vector index
    4. Loading chunk metadata
    5. Retrieving relevant chunks for a user query
    6. Reranking retrieved chunks
    7. Building a formatted context block for prompting
    """

    def __init__(
        self,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        top_k=TOP_K,
        final_k=FINAL_K,
        embed_model_name=EMBED_MODEL,
        rerank_model_name=RERANK_MODEL,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        index_path : str
            Path to FAISS index file.

        meta_path : str
            Path to metadata pickle file.

        top_k : int
            Number of chunks to retrieve from FAISS before reranking.

        final_k : int
            Number of chunks to keep after reranking.

        embed_model_name : str
            SentenceTransformer embedding model.

        rerank_model_name : str
            CrossEncoder reranker model.

        verbose : bool
            Whether to print loading progress to stdout.
        """

        self.index_path = index_path
        self.meta_path = meta_path
        self.top_k = top_k
        self.final_k = final_k
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name
        self.verbose = verbose

        self.embed_model = None
        self.reranker = None
        self.index = None
        self.metadata = None

        self._load_components(verbose=self.verbose)

    # ==========================================================
    # INTERNAL LOADER
    # ==========================================================

    def _load_components(self, verbose: bool = False):
        """
        Load all retrieval-related components silently or with clean status:
        - embedding model
        - reranker
        - FAISS index
        - metadata
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            if not verbose:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

            self.embed_model = SentenceTransformer(
                self.embed_model_name
            )

            self.reranker = CrossEncoder(
                self.rerank_model_name
            )

            self.index = faiss.read_index(
                self.index_path
            )

            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        finally:
            if not verbose:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        if verbose:
            print("[+] RAG Neural Engine ready.")


    # ==========================================================
    # QUERY EMBEDDING
    # ==========================================================

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Convert a user query into a float32 embedding for FAISS.
        """

        query_embedding = self.embed_model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        return query_embedding

    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    def retrieve(self, query: str, min_score: float = -2.5):
        """
        Retrieve top documents for a query and rerank them, filtering out negative irrelevant logits.

        Returns
        -------
        list[tuple[dict, float]]
            List of tuples:
            [
                (doc_metadata_dict, reranker_score),
                ...
            ]
        """

        # --------------------------
        # Embed query
        # --------------------------
        query_embedding = self._embed_query(query)

        # --------------------------
        # Search FAISS
        # --------------------------
        distances, indices = self.index.search(
            query_embedding,
            self.top_k
        )

        # --------------------------
        # Collect candidate docs & compute normalized similarity (0..1)
        # --------------------------
        similarities = 1.0 / (1.0 + distances[0])
        candidates = []

        for idx, dist, sim in zip(indices[0], distances[0], similarities):
            if idx < 0:
                continue

            # safety check
            if idx < len(self.metadata):
                meta = dict(self.metadata[idx])
                meta["_faiss_dist"] = float(dist)
                meta["_score"] = float(sim)  # normalized 0..1, higher is better
                candidates.append((meta, float(sim)))

        # If nothing found
        if not candidates:
            return []

        # --------------------------
        # Build query-doc pairs
        # --------------------------
        pairs = [
            [query, doc.get("text", "")]
            for doc, _ in candidates
        ]

        # --------------------------
        # Rerank
        # --------------------------
        scores = self.reranker.predict(
            pairs
        )

        ranked = []
        for (doc, sim), rerank_score in zip(candidates, scores):
            doc["_rerank_score"] = float(rerank_score)
            if float(rerank_score) >= min_score and sim >= 0.25:
                ranked.append((doc, float(sim)))

        ranked.sort(key=lambda x: x[0].get("_rerank_score", x[1]), reverse=True)
        return ranked[:self.final_k]

    # ==========================================================
    # CONTEXT BUILDER
    # ==========================================================

    def build_context(self, ranked_docs, min_sim: float = 0.35):
        """
        Convert reranked docs into one formatted context string.
        Filters out low-similarity chunks below min_sim (default 0.35).

        Parameters
        ----------
        ranked_docs : list[tuple[dict, float]]
            Output from retrieve()
        min_sim : float
            Minimum similarity threshold (0..1) to include in context.

        Returns
        -------
        str
            Formatted context block for the LLM prompt.
        """

        if not ranked_docs:
            return ""

        context_parts = []

        for doc, score in ranked_docs:
            sim = doc.get("_score", score)
            if sim < min_sim:
                continue

            source = doc.get(
                "filename",
                doc.get("source", "Unknown")
            )

            chunk_id = doc.get(
                "chunk_id",
                "N/A"
            )

            text = doc.get(
                "text",
                ""
            )

            block = (
                f"[SOURCE: {source}]\n"
                f"[CHUNK: {chunk_id}]\n"
                f"[SIMILARITY: {sim:.4f}]\n"
                f"{text}"
            )

            context_parts.append(block)

        return "\n\n".join(context_parts)

    # ==========================================================
    # OPTIONAL DEBUG INFO
    # ==========================================================

    def info(self) -> dict:
        """
        Return engine configuration for debugging.
        """
        return {
            "index_path": self.index_path,
            "meta_path": self.meta_path,
            "top_k": self.top_k,
            "final_k": self.final_k,
            "embed_model": self.embed_model_name,
            "rerank_model": self.rerank_model_name,
            "metadata_count": len(self.metadata) if self.metadata else 0
        }


def get_global_rag_engine(
    index_path=INDEX_PATH,
    meta_path=META_PATH,
    top_k=TOP_K,
    final_k=FINAL_K,
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
        except Exception:
            _GLOBAL_RAG_ENGINE = False

    return _GLOBAL_RAG_ENGINE if _GLOBAL_RAG_ENGINE is not False else None