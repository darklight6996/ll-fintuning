"""
core/yaml_search.py

Purpose
-------
Find the most relevant assessment templates for a user's request using semantic search.

Responsibilities
----------------
1. Convert user request/question into semantic vector embeddings.
2. Embed loaded YAML assessment templates (name, category, description, tags, steps).
3. Perform similarity search and return ranked YAML templates matching user intent.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from core.yaml_registry import YAMLRegistry
from core.config import EMBED_MODEL


class YAMLSearch:
    """
    Semantic search engine for YAML assessment templates.
    """

    def __init__(
        self,
        registry: YAMLRegistry,
        model_name: str = EMBED_MODEL
    ):
        """
        Initialize YAML search with a registry and embedding model.

        Args:
            registry: Populated YAMLRegistry instance
            model_name: SentenceTransformer model name
        """
        self.registry = registry
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        self._template_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self.indexed = False
        
        # Build index automatically if registry is loaded
        if self.registry and self.registry.loaded:
            self.build_index()

    def _template_to_text(self, template: Dict[str, Any]) -> str:
        """
        Convert a template into a searchable text chunk combining key fields.
        """
        parts = [
            f"Title: {template.get('name', '')}",
            f"Category: {template.get('category', '')}",
            f"Subcategory: {template.get('subcategory', '')}",
            f"Description: {template.get('description', '')}",
            f"Tags: {', '.join(template.get('tags', []))}",
        ]

        steps = template.get("steps", [])
        step_descriptions = []
        for step in steps:
            if isinstance(step, dict):
                action = step.get("action", "")
                phase = step.get("phase", "")
                if action:
                    step_descriptions.append(f"[{phase}] {action}")

        if step_descriptions:
            parts.append(f"Steps: {' | '.join(step_descriptions)}")

        return "\n".join(parts)

    def build_index(self):
        """
        Compute embeddings for all templates currently loaded in the registry.
        """
        templates = self.registry.all()
        if not templates:
            self._template_ids = []
            self._embeddings = None
            self.indexed = True
            return

        self._template_ids = []
        texts = []

        for tmpl in templates:
            tid = tmpl.get("id")
            if tid:
                self._template_ids.append(tid)
                texts.append(self._template_to_text(tmpl))

        if texts:
            raw_embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self._embeddings = np.array(raw_embeddings, dtype=np.float32)
        else:
            self._embeddings = None

        self.indexed = True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for assessment templates matching the user prompt.

        Args:
            query: User's intent/question (e.g. "Assess Azure tenant for privilege escalation")
            top_k: Number of relevant templates to return

        Returns:
            List of matching template dicts with an added '_score' key.
        """
        if not self.indexed:
            self.build_index()

        if self._embeddings is None or len(self._template_ids) == 0:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Cosine similarity for normalized vectors is simple dot product
        scores = np.dot(self._embeddings, query_vec)
        
        # Sort descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            tid = self._template_ids[idx]
            template = self.registry.get(tid)
            if template:
                res = dict(template)
                res["_score"] = score
                results.append(res)

        return results
