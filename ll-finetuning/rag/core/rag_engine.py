# ==========================================================
# core/rag_engine.py
# Retrieval + Reranking Engine for RAG
# ==========================================================

# ==============================
# IMPORTS
# ==============================

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
        rerank_model_name=RERANK_MODEL
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
        """

        self.index_path = index_path
        self.meta_path = meta_path
        self.top_k = top_k
        self.final_k = final_k
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name

        self.embed_model = None
        self.reranker = None
        self.index = None
        self.metadata = None

        self._load_components()

    # ==========================================================
    # INTERNAL LOADER
    # ==========================================================

    def _load_components(self):
        """
        Load all retrieval-related components:
        - embedding model
        - reranker
        - FAISS index
        - metadata
        """

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(
            self.embed_model_name
        )

        print("Loading reranker...")
        self.reranker = CrossEncoder(
            self.rerank_model_name
        )

        print("Loading FAISS index...")
        self.index = faiss.read_index(
            self.index_path
        )

        print("Loading metadata...")
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("RAG Engine loaded successfully.")

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

    def retrieve(self, query: str):
        """
        Retrieve top documents for a query and rerank them.

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
        # Collect candidate docs
        # --------------------------
        candidates = []

        for idx in indices[0]:
            if idx == -1:
                continue

            # safety check
            if idx < len(self.metadata):
                candidates.append(
                    self.metadata[idx]
                )

        # If nothing found
        if not candidates:
            return []

        # --------------------------
        # Build query-doc pairs
        # --------------------------
        pairs = [
            [query, doc.get("text", "")]
            for doc in candidates
        ]

        # --------------------------
        # Rerank
        # --------------------------
        scores = self.reranker.predict(
            pairs
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.final_k]

    # ==========================================================
    # CONTEXT BUILDER
    # ==========================================================

    def build_context(self, ranked_docs):
        """
        Convert reranked docs into one formatted context string.

        Parameters
        ----------
        ranked_docs : list[tuple[dict, float]]
            Output from retrieve()

        Returns
        -------
        str
            Formatted context block for the LLM prompt.
        """

        if not ranked_docs:
            return "No relevant context retrieved."

        context_parts = []

        for doc, score in ranked_docs:
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
                f"[SCORE: {score:.4f}]\n"
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