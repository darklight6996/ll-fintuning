"""
core/test_embedding_service.py

Unit tests for Phase 2: Shared Embedding Service.
Validates:
1. Singleton pattern guarantee (only one instance per process).
2. embed() output shape, dtype (float32), and L2 normalization.
3. embed_many() batch output shape (N, dim) and dtype.
4. Normalize helper method.
5. In-memory LRU query cache (hit/miss/clear).
6. Thread safety across concurrent invocations.
7. Backward compatibility (.encode() and .model access).
"""

import sys
import unittest
from pathlib import Path
import threading
import numpy as np

# Resolve system paths to include rag/
RAG_DIR = Path(__file__).resolve().parent.parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from core.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    EmbeddingCache,
)


class TestEmbeddingService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = get_embedding_service(verbose=False)

    def test_01_singleton_identity(self):
        """Ensure get_embedding_service always returns the exact same instance."""
        svc1 = get_embedding_service()
        svc2 = get_embedding_service()
        self.assertIs(svc1, svc2)
        self.assertIs(svc1, self.service)

    def test_02_embed_single_text(self):
        """Ensure embed() returns a 1D float32 array with unit norm when normalized."""
        text = "How to perform privilege escalation in Linux?"
        vec = self.service.embed(text, normalize=True)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(vec.ndim, 1)
        self.assertEqual(len(vec), self.service.dimension)

        # Check unit L2 norm
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_03_embed_many_batch(self):
        """Ensure embed_many() returns a 2D float32 array of shape (N, dim)."""
        texts = [
            "Assess Azure tenant role assignments",
            "Active Directory Kerberoasting attacks",
            "Metasploit payload generation"
        ]
        matrix = self.service.embed_many(texts, normalize=True)

        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (len(texts), self.service.dimension))

        # Check unit L2 norms across all rows
        norms = np.linalg.norm(matrix, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=4)

    def test_04_normalize_helper(self):
        """Test the static normalize method on arbitrary 1D and 2D arrays."""
        # 1D test
        raw_1d = np.array([3.0, 4.0], dtype=np.float32)
        norm_1d = EmbeddingService.normalize(raw_1d)
        self.assertEqual(norm_1d.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(norm_1d)), 1.0, places=5)
        self.assertTrue(np.allclose(norm_1d, [0.6, 0.8]))

        # 2D test
        raw_2d = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
        norm_2d = EmbeddingService.normalize(raw_2d)
        self.assertEqual(norm_2d.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(norm_2d[0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(norm_2d[1])), 1.0, places=5)

    def test_05_cache_behavior(self):
        """Ensure query cache returns identical objects and increments properly."""
        cache = EmbeddingCache(max_size=10, enabled=True)
        test_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        self.assertIsNone(cache.get("query_a"))
        cache.put("query_a", test_vec)
        self.assertEqual(len(cache), 1)

        retrieved = cache.get("query_a")
        self.assertIsNotNone(retrieved)
        self.assertTrue(np.allclose(retrieved, test_vec))

        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get("query_a"))

    def test_06_thread_safety(self):
        """Test concurrent embedding requests across multiple threads."""
        results = {}
        errors = []

        def worker(thread_id, query):
            try:
                vec = self.service.embed(query, normalize=True)
                results[thread_id] = vec
            except Exception as e:
                errors.append(e)

        threads = []
        queries = [
            f"Concurrent security query #{i} testing thread safety."
            for i in range(10)
        ]

        for i, q in enumerate(queries):
            t = threading.Thread(target=worker, args=(i, q))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
        self.assertEqual(len(results), 10)
        for vec in results.values():
            self.assertEqual(vec.dtype, np.float32)
            self.assertEqual(len(vec), self.service.dimension)

    def test_07_backward_compatibility(self):
        """Ensure encode() alias and .model property function as expected."""
        # .encode() on single string
        res_str = self.service.encode("single test string")
        self.assertEqual(res_str.ndim, 2)
        self.assertEqual(res_str.shape, (1, self.service.dimension))

        # .encode() on list of strings
        res_list = self.service.encode(["str 1", "str 2"])
        self.assertEqual(res_list.shape, (2, self.service.dimension))

        # .model access
        self.assertIsNotNone(self.service.model)


if __name__ == "__main__":
    unittest.main()
