"""
core/retrieval package

Decoupled retrieval subsystem for Cybersecurity RAG:
- VectorStore: Abstraction & adapters for FAISS, HNSW, Qdrant, Chroma
- Retriever: Query embedding + vector similarity search + chunk filtering
- Reranker: Neural cross-encoder reranking
- ContextBuilder: Formatted prompt context compilation
"""

from core.retrieval.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    HNSWVectorStore,
    QdrantVectorStore,
    ChromaVectorStore,
    create_vector_store
)
from core.retrieval.retriever import Retriever, ChunkFilter
from core.retrieval.reranker import Reranker
from core.retrieval.context_builder import RAGContextBuilder

__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
    "HNSWVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "create_vector_store",
    "Retriever",
    "ChunkFilter",
    "Reranker",
    "RAGContextBuilder",
]
