"""
core/builders/context_builder.py

Assembles multi-source context (live session intelligence, RAG documents, and YAML tactics)
into a unified ContextPackage.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from core.services.rag_service import get_rag_service
from core.services.yaml_service import get_yaml_service


@dataclass
class ContextPackage:
    """Encapsulates all contextual elements assembled for answering a query."""
    intel: Dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    yaml_tactics: List[str] = field(default_factory=list)


class ContextBuilder:
    """
    Coordinates context extraction across session state, knowledge base (RAG),
    and procedural tactics (YAML scenarios).
    """

    def __init__(
        self,
        rag_svc: Optional[Any] = None,
        yaml_svc: Optional[Any] = None,
        intelligence_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        self._rag_svc = rag_svc
        self._yaml_svc = yaml_svc
        self._intel_extractor = intelligence_extractor

    def _get_rag(self):
        if self._rag_svc is None:
            self._rag_svc = get_rag_service() or False
        return self._rag_svc if self._rag_svc is not False else None

    def _get_yaml(self):
        if self._yaml_svc is None:
            self._yaml_svc = get_yaml_service() or False
        return self._yaml_svc if self._yaml_svc is not False else None

    def build(
        self,
        query: str,
        state: Optional[Dict[str, Any]] = None,
        intel: Optional[Dict[str, Any]] = None
    ) -> ContextPackage:
        """
        Build a complete ContextPackage for the given query and session state.
        """
        clean_query = query.strip()
        state = state or {}

        # 1. Session intelligence
        if intel is None:
            if self._intel_extractor:
                intel = self._intel_extractor(state)
            else:
                try:
                    from core.intelligence import get_intelligence_extractor
                    intel = get_intelligence_extractor().extract_dict(state)
                except Exception:
                    intel = {}

        # 2. RAG retrieval
        rag_context = ""
        rag = self._get_rag()
        if rag:
            try:
                retrieved = rag.retrieve(clean_query)
                rag_context = rag.build_context(retrieved)
            except Exception:
                pass

        # 3. YAML procedural tactics
        yaml_tactics = []
        yaml_search = self._get_yaml()
        if yaml_search:
            try:
                tmpls = yaml_search.search(clean_query, top_k=2)
                for t in tmpls:
                    yaml_tactics.append(
                        f"* [{t.get('category', 'Tactic')}] {t.get('name')}: {t.get('description', '')[:120]}"
                    )
            except Exception:
                pass

        return ContextPackage(
            intel=intel,
            rag_context=rag_context,
            yaml_tactics=yaml_tactics
        )
