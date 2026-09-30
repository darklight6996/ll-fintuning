"""
core/yaml_search.py

Purpose
-------
Find the most relevant assessment templates for a user's request using persistent vector search.

Responsibilities
----------------
1. Load persistent FAISS index and metadata on startup (sub-10ms, zero re-embeddings).
2. Automatically sync with YAML scenarios using YAMLIndexer (detects added/modified/deleted files).
3. Embed user search queries using shared EmbeddingService.
4. Perform cosine similarity retrieval and return ranked YAML templates matching user intent.
5. Provide backward compatibility for existing callers and registries.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.config import EMBED_MODEL, SCENARIOS_DIR, YAML_INDEX_DIR
from core.yaml_registry import YAMLRegistry
from core.yaml_indexer import YAMLIndexer
from core.services.embedding_service import get_embedding_service, EmbeddingService


class YAMLSearch:
    """
    High-performance semantic search engine for YAML assessment templates backed by persistent FAISS index.
    """

    def __init__(
        self,
        registry: Optional[YAMLRegistry] = None,
        model_name: str = EMBED_MODEL,
        yaml_root: str = SCENARIOS_DIR,
        output_dir: str = YAML_INDEX_DIR,
        indexer: Optional[YAMLIndexer] = None,
        auto_sync: bool = True,
        verbose: bool = False
    ):
        """
        Initialize YAML search with persistent index backend.

        Args:
            registry: Optional populated YAMLRegistry instance (for backward compatibility).
            model_name: Embedding model name.
            yaml_root: Path to directory containing YAML scenario files.
            output_dir: Path to directory storing yaml.index and yaml_meta.pkl.
            indexer: Optional existing YAMLIndexer instance.
            auto_sync: If True, checks for modified/added YAML files and updates index incrementally.
            verbose: If True, prints status logs.
        """
        self.registry = registry
        self.model_name = model_name
        self.embedding_service = get_embedding_service(model_name=model_name)
        # Expose self.model for backward compatibility
        self.model = self.embedding_service
        self.verbose = verbose

        self.indexer = indexer or YAMLIndexer(
            yaml_root=yaml_root,
            output_dir=output_dir,
            embedding_model=model_name,
            verbose=verbose
        )

        self._index: Optional[faiss.Index] = None
        self._meta_store: Dict[str, Any] = {}
        self._templates: List[Dict[str, Any]] = []
        self._template_ids: List[str] = []
        self._templates_by_id: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Optional[np.ndarray] = None
        self.indexed: bool = False

        if auto_sync:
            self.sync()

    # ======================================================
    # INDEX SYNCHRONIZATION & LOADING
    # ======================================================

    def sync(self, force_rebuild: bool = False) -> None:
        """
        Sync or load the persistent FAISS index and metadata store.
        """
        self._index, self._meta_store = self.indexer.sync_index(force_rebuild=force_rebuild)
        self._templates = self._meta_store.get("templates", [])
        self._template_ids = self._meta_store.get("template_ids", [])
        self._embeddings = self._meta_store.get("embeddings")
        self._templates_by_id = {
            tmpl.get("id"): tmpl
            for tmpl in self._templates
            if tmpl and tmpl.get("id")
        }
        self.indexed = self._index is not None and len(self._templates) > 0

    def build_index(self) -> None:
        """Backward-compatible rebuild trigger."""
        self.sync(force_rebuild=True)

    # ======================================================
    # SEARCH & RETRIEVAL
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for assessment templates matching the user prompt.

        Args:
            query: User intent/question (e.g. "Assess Azure tenant for privilege escalation")
            top_k: Number of relevant templates to return.
            min_score: Optional minimum cosine similarity score threshold (e.g. 0.2).
            category: Optional filter by category name.
            tag: Optional filter by tag name.

        Returns:
            List of matching template dicts with an added '_score' key.
        """
        if not self.indexed:
            self.sync()

        if self._index is None or not self._templates:
            return []

        # Convert query to normalized float32 vector
        query_vec = self.embedding_service.embed(query, normalize=True)
        query_matrix = query_vec.reshape(1, -1)

        # Retrieve candidates from FAISS
        fetch_k = min(len(self._templates), top_k * 3 if (category or tag or min_score) else top_k)
        if fetch_k <= 0:
            return []

        scores, indices = self._index.search(query_matrix, fetch_k)

        results = []
        for dist, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self._templates):
                continue

            score = float(dist)
            if min_score is not None and score < min_score:
                continue

            template = self._templates[idx]
            if not template:
                continue

            # Apply category filter if requested
            if category:
                tmpl_cat = str(template.get("category", "")).lower()
                if category.lower() not in tmpl_cat:
                    continue

            # Apply tag filter if requested
            if tag:
                tmpl_tags = [str(t).lower() for t in template.get("tags", [])]
                if tag.lower() not in tmpl_tags:
                    continue

            res = dict(template)
            res["_score"] = score
            results.append(res)

            if len(results) >= top_k:
                break

        return results

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID directly from metadata cache."""
        if not self.indexed:
            self.sync()
        return self._templates_by_id.get(template_id)

    @property
    def templates(self) -> List[Dict[str, Any]]:
        """Return all loaded templates."""
        if not self.indexed:
            self.sync()
        return self._templates

    def __len__(self) -> int:
        if not self.indexed:
            self.sync()
        return len(self._templates)
