"""
core/prompts/knowledge_builder.py

KnowledgeBuilder compresses and cleans retrieved RAG documentation snippets,
removing duplicate information and enforcing character budget limits.
"""

from typing import List, Tuple, Dict, Any, Optional, Union


class KnowledgeBuilder:
    """
    Constructs the TRUSTED KNOWLEDGE (RAG) section of the prompt.
    Deduplicates overlapping knowledge passages and preserves clean source metadata.
    """

    def deduplicate_snippets(self, raw_text: str) -> str:
        """
        Deduplicate identical blocks or repeated chunk sentences in raw RAG context.
        """
        if not raw_text:
            return ""

        blocks = raw_text.split("\n\n")
        seen_texts = set()
        unique_blocks = []

        for b in blocks:
            b_clean = b.strip()
            if not b_clean:
                continue

            # Key on content after headers
            lines = b_clean.split("\n")
            content_lines = [l for l in lines if not l.startswith("[SOURCE") and not l.startswith("[CHUNK") and not l.startswith("[SIMILARITY")]
            content_key = " ".join(content_lines).strip()[:150]

            if content_key and content_key in seen_texts:
                continue

            if content_key:
                seen_texts.add(content_key)
            unique_blocks.append(b_clean)

        return "\n\n".join(unique_blocks)

    def build(
        self,
        rag_context: Optional[str] = None,
        max_chars: int = 1500
    ) -> str:
        """
        Build and truncate trusted knowledge block within allocated budget.
        """
        if not rag_context or not rag_context.strip():
            return "(no relevant knowledge documents retrieved)"

        cleaned = self.deduplicate_snippets(rag_context)

        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip() + "\n... [knowledge truncated to budget]"

        return cleaned
