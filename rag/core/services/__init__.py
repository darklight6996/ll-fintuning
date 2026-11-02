"""
core/services/__init__.py

Services layer — singleton accessors for all shared infrastructure.
All services are lazily initialized on first access and then reused for
the lifetime of the process, eliminating duplicate model loading.

Usage:
    from core.services import get_embedding_service, get_llm_service
    emb = get_embedding_service()
    llm = get_llm_service()
"""

from core.services.embedding_service import get_embedding_service
from core.services.rerank_service import get_rerank_service
from core.services.llm_service import get_llm_service
from core.services.rag_service import get_rag_service
from core.services.yaml_service import get_yaml_service
from core.services.memory_service import get_memory

__all__ = [
    "get_embedding_service",
    "get_rerank_service",
    "get_llm_service",
    "get_rag_service",
    "get_yaml_service",
    "get_memory",
]
