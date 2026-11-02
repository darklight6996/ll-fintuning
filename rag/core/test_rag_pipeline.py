"""
test_rag_pipeline.py

Unit test suite verifying Phase 4 RAG Refactor:
1. VectorStore abstraction and FAISS adapter.
2. Config-driven vector store swapping via create_vector_store().
3. ChunkFilter similarity and metadata enforcement.
4. Dependency Injection in Retriever, Reranker, and RAGContextBuilder.
5. Backward compatibility of RAGEngine.
"""

import os
import sys
import tempfile
import pickle
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.retrieval.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    create_vector_store
)
from core.retrieval.retriever import Retriever, ChunkFilter
from core.retrieval.reranker import Reranker
from core.retrieval.context_builder import RAGContextBuilder
from core.rag_engine import RAGEngine


# ==========================================================
# 1. VECTOR STORE & FAISS ADAPTER TESTS
# ==========================================================

def test_faiss_vector_store_in_memory():
    """Verify FAISS vector store can add and search vectors in memory."""
    store = FAISSVectorStore(load_from_disk=False, dim=4)
    assert not store.is_ready()
    assert store.count() == 0

    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    metadata = [
        {"id": 1, "text": "doc 1", "source": "s1.md"},
        {"id": 2, "text": "doc 2", "source": "s2.md"},
        {"id": 3, "text": "doc 3", "source": "s3.md"}
    ]

    store.add(vectors, metadata)
    assert store.is_ready()
    assert store.count() == 3

    # Query identical to vector 1
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)
    assert len(results) == 2
    top_doc, top_sim = results[0]
    assert top_doc["id"] == 1
    assert top_sim == pytest.approx(1.0, abs=1e-4)
    assert "_score" in top_doc
    assert "_faiss_dist" in top_doc


def test_vector_store_factory_switching():
    """Verify create_vector_store accurately routes backend requests."""
    faiss_store = create_vector_store(backend="faiss", index_path=None, meta_path=None)
    assert isinstance(faiss_store, FAISSVectorStore)

    with pytest.raises(ValueError, match="Unsupported vector store backend"):
        create_vector_store(backend="unsupported_db")


# ==========================================================
# 2. CHUNK FILTER TESTS
# ==========================================================

def test_chunk_filter():
    """Verify similarity cutoff and metadata key validation."""
    filter_obj = ChunkFilter(default_min_sim=0.30)
    candidates = [
        ({"id": 1, "source": "a.md"}, 0.50),
        ({"id": 2, "source": "b.md"}, 0.20),
        ({"id": 3}, 0.40)  # Missing source
    ]

    # Filter with min_sim=0.30
    res = filter_obj.filter(candidates, min_sim=0.30)
    assert len(res) == 2
    assert res[0][0]["id"] == 1
    assert res[1][0]["id"] == 3

    # Filter requiring "source" key
    res_keys = filter_obj.filter(candidates, min_sim=0.10, required_keys=["source"])
    assert len(res_keys) == 2
    assert [r[0]["id"] for r in res_keys] == [1, 2]


# ==========================================================
# 3. RETRIEVER DEPENDENCY INJECTION TESTS
# ==========================================================

def test_retriever_with_mocks():
    """Verify Retriever coordinates embedding, store search, and filtering via DI."""
    mock_store = MagicMock(spec=BaseVectorStore)
    mock_store.is_ready.return_value = True
    mock_store.search.return_value = [
        ({"id": 10, "text": "SUID binary found", "source": "ctf.md"}, 0.85),
        ({"id": 20, "text": "Irrelevant log entry", "source": "sys.log"}, 0.15)
    ]

    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_many.return_value = np.zeros((1, 384), dtype=np.float32)

    retriever = Retriever(
        vector_store=mock_store,
        embedding_service=mock_embed_svc,
        chunk_filter=ChunkFilter(default_min_sim=0.30)
    )

    results = retriever.retrieve("find SUID binaries", top_k=5, min_sim=0.30)
    assert len(results) == 1
    assert results[0][0]["id"] == 10
    mock_embed_svc.embed_many.assert_called_once_with(["find SUID binaries"])
    mock_store.search.assert_called_once()


# ==========================================================
# 4. RERANKER DEPENDENCY INJECTION TESTS
# ==========================================================

def test_reranker_with_mock_service():
    """Verify Reranker uses injected RerankService to score and order results."""
    mock_rerank_svc = MagicMock()
    mock_rerank_svc.is_available = True
    # Return scores for 2 candidates: doc 1 gets 1.5, doc 2 gets -3.0
    mock_rerank_svc.predict.return_value = np.array([1.5, -3.0])

    reranker = Reranker(rerank_service=mock_rerank_svc)
    candidates = [
        ({"id": 1, "text": "Kernel exploit details"}, 0.70),
        ({"id": 2, "text": "Unrelated network config"}, 0.65)
    ]

    # min_score = -2.5 drops doc 2
    ranked = reranker.rerank("privilege escalation", candidates, min_score=-2.5, final_k=2)
    assert len(ranked) == 1
    assert ranked[0][0]["id"] == 1
    assert ranked[0][0]["_rerank_score"] == 1.5


# ==========================================================
# 5. RAG CONTEXT BUILDER TESTS
# ==========================================================

def test_rag_context_builder():
    """Verify RAGContextBuilder formatting and similarity thresholds."""
    builder = RAGContextBuilder(default_min_sim=0.35)
    docs = [
        ({"filename": "playbook.md", "chunk_id": "c1", "text": "Run id and sudo -l", "_score": 0.82}, 0.82),
        ({"filename": "notes.md", "chunk_id": "c2", "text": "Low score note", "_score": 0.20}, 0.20)
    ]

    ctx = builder.build(docs, min_sim=0.35)
    assert "[SOURCE: playbook.md]" in ctx
    assert "[CHUNK: c1]" in ctx
    assert "[SIMILARITY: 0.8200]" in ctx
    assert "Run id and sudo -l" in ctx
    assert "Low score note" not in ctx


# ==========================================================
# 6. RAG ENGINE BACKWARD COMPATIBILITY
# ==========================================================

def test_rag_engine_backward_compatibility():
    """Verify RAGEngine coordinates all injected components with existing API."""
    mock_store = MagicMock(spec=BaseVectorStore)
    mock_store.is_ready.return_value = True
    mock_store.count.return_value = 100
    mock_store.search.return_value = [
        ({"filename": "ad.md", "chunk_id": "c9", "text": "BloodHound recon", "_score": 0.90}, 0.90)
    ]

    mock_retriever = Retriever(vector_store=mock_store, chunk_filter=ChunkFilter(default_min_sim=0.20))
    mock_retriever.embed_query = MagicMock(return_value=np.zeros((1, 4), dtype=np.float32))

    mock_rerank_svc = MagicMock()
    mock_rerank_svc.is_available = True
    mock_rerank_svc.predict.return_value = np.array([2.8])
    mock_reranker = Reranker(rerank_service=mock_rerank_svc)

    engine = RAGEngine(
        index_path="dummy.index",
        meta_path="dummy.pkl",
        vector_store=mock_store,
        retriever=mock_retriever,
        reranker=mock_reranker,
        verbose=False
    )

    info = engine.info()
    assert info["metadata_count"] == 100
    assert info["is_ready"] is True

    # Retrieve
    results = engine.retrieve("Bloodhound domain admin", min_score=0.0)
    assert len(results) == 1
    assert results[0][0]["filename"] == "ad.md"

    # Build context
    ctx = engine.build_context(results)
    assert "[SOURCE: ad.md]" in ctx
    assert "BloodHound recon" in ctx
