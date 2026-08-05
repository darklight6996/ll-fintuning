"""
core/assessment_router.py

Purpose
-------
Traffic controller deciding where information should come from before answering.

Responsibilities
----------------
1. Coordinate query routing between Document RAG Engine and YAML Search Engine.
2. Retrieve documentation context (knowledge RAG).
3. Retrieve procedural context (YAML templates RAG).
4. Combine contexts into a unified payload for LLM and Planner.
"""

from typing import Dict, Any, List, Optional
from core.rag_engine import RAGEngine
from core.yaml_search import YAMLSearch


class MergedContext:
    """
    Container for combined documentation and procedural context.
    """

    def __init__(
        self,
        query: str,
        documents: List[tuple],
        doc_context_str: str,
        yaml_templates: List[Dict[str, Any]]
    ):
        self.query = query
        self.documents = documents
        self.doc_context_str = doc_context_str
        self.yaml_templates = yaml_templates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "document_count": len(self.documents),
            "doc_context": self.doc_context_str,
            "yaml_templates": self.yaml_templates,
            "yaml_ids": [t.get("id") for t in self.yaml_templates]
        }

    def format_combined_prompt_context(self) -> str:
        """
        Format both documentation context and assessment YAML templates into a single string.
        """
        parts = ["=== DOCUMENTATION CONTEXT ===", self.doc_context_str, ""]

        parts.append("=== RELEVANT ASSESSMENT PROCEDURES ===")
        if not self.yaml_templates:
            parts.append("No specific assessment procedures retrieved.")
        else:
            for tmpl in self.yaml_templates:
                parts.append(f"Template ID: {tmpl.get('id')}")
                parts.append(f"Name: {tmpl.get('name')}")
                parts.append(f"Category: {tmpl.get('category')} / {tmpl.get('subcategory')}")
                parts.append(f"Description: {tmpl.get('description', '').strip()}")
                parts.append("---")

        return "\n".join(parts)


class AssessmentRouter:
    """
    Decides and routes retrieval requests to both document RAG and YAML procedure search.
    """

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        yaml_search: Optional[YAMLSearch] = None
    ):
        self.rag_engine = rag_engine
        self.yaml_search = yaml_search

    def route(
        self,
        query: str,
        top_k_docs: int = 3,
        top_k_yamls: int = 3
    ) -> MergedContext:
        """
        Retrieve documentation and YAML assessment procedures for a user request.

        Args:
            query: User prompt/question
            top_k_docs: Number of document chunks to retrieve
            top_k_yamls: Number of assessment templates to retrieve

        Returns:
            MergedContext containing both doc results and YAML templates.
        """
        # 1. Document Search
        docs = []
        doc_context_str = "Document RAG Engine unavailable."
        if self.rag_engine:
            try:
                # Set temporary retrieve limit if provided
                old_k = self.rag_engine.final_k
                self.rag_engine.final_k = top_k_docs
                docs = self.rag_engine.retrieve(query)
                doc_context_str = self.rag_engine.build_context(docs)
                self.rag_engine.final_k = old_k
            except Exception as e:
                doc_context_str = f"Error during document retrieval: {str(e)}"

        # 2. YAML Search
        yaml_templates = []
        if self.yaml_search:
            try:
                yaml_templates = self.yaml_search.search(query, top_k=top_k_yamls)
            except Exception as e:
                print(f"[AssessmentRouter] Error during YAML search: {e}")

        return MergedContext(
            query=query,
            documents=docs,
            doc_context_str=doc_context_str,
            yaml_templates=yaml_templates
        )
