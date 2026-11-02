"""
core/retrieval/context_builder.py

Formats retrieved and reranked knowledge chunks into clean, prompt-ready context blocks.
Filters low-similarity or malformed chunks.
"""

from typing import List, Tuple, Dict, Any, Optional


class RAGContextBuilder:
    """
    Transforms reranked document records into structured text blocks for LLM prompting.
    """

    def __init__(self, default_min_sim: float = 0.35):
        self.default_min_sim = default_min_sim

    def build(
        self,
        ranked_docs: List[Tuple[Dict[str, Any], float]],
        min_sim: Optional[float] = None,
        max_chars: Optional[int] = None
    ) -> str:
        """
        Convert reranked docs into a formatted context string.

        Parameters
        ----------
        ranked_docs : List[Tuple[Dict[str, Any], float]]
            List of (doc_metadata, similarity) from reranker.
        min_sim : Optional[float]
            Minimum similarity threshold (0..1) to include in context.
        max_chars : Optional[int]
            Character limit for overall context block.

        Returns
        -------
        str
            Formatted context block ready for inclusion in prompts.
        """
        if not ranked_docs:
            return ""

        threshold = self.default_min_sim if min_sim is None else min_sim
        context_parts = []
        total_len = 0

        for doc, score in ranked_docs:
            sim = doc.get("_score", score)
            if sim < threshold:
                continue

            source = doc.get("filename", doc.get("source", "Unknown"))
            chunk_id = doc.get("chunk_id", "N/A")
            text = doc.get("text", "").strip()

            block = (
                f"[SOURCE: {source}]\n"
                f"[CHUNK: {chunk_id}]\n"
                f"[SIMILARITY: {sim:.4f}]\n"
                f"{text}"
            )

            if max_chars and (total_len + len(block) > max_chars):
                remaining = max_chars - total_len
                if remaining > 100:
                    context_parts.append(block[:remaining] + "... [truncated]")
                break

            context_parts.append(block)
            total_len += len(block) + 2  # account for "\n\n"

        return "\n\n".join(context_parts)
