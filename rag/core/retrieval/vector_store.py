"""
core/retrieval/vector_store.py

Vector store abstraction layer supporting pluggable database backends:
- FAISS (default, production verified)
- HNSW (hnswlib adapter)
- Qdrant (qdrant_client adapter)
- Chroma (chromadb adapter)

Allows swapping vector storage engines through configuration alone without
altering downstream retrieval or prompting logic.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import pickle
import numpy as np

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.config import (
    INDEX_PATH,
    META_PATH,
    VECTOR_STORE_BACKEND,
)


class BaseVectorStore(ABC):
    """
    Abstract contract for vector store implementations.
    """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute vector similarity search.

        Parameters
        ----------
        query_vector : np.ndarray
            1D or 2D float32 numpy array representing the query embedding.
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of (metadata_dict, similarity_score) sorted by descending similarity.
            Similarity scores are normalized in [0.0, 1.0].
        """
        pass

    @abstractmethod
    def add(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add embedding vectors and their corresponding metadata records.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in store."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if index and metadata are loaded and ready for search."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store adapter.
    Uses L2 distance normalized to [0..1] similarity: 1.0 / (1.0 + distance).
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        dim: int = 384,
        load_from_disk: bool = True
    ):
        self.dim = dim
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        if load_from_disk:
            self.index_path = index_path if index_path is not None else INDEX_PATH
            self.meta_path = meta_path if meta_path is not None else META_PATH
            if self.index_path and os.path.exists(self.index_path) and \
               self.meta_path and os.path.exists(self.meta_path):
                self.load(self.index_path, self.meta_path)
        else:
            self.index_path = index_path
            self.meta_path = meta_path

    def load(self, index_path: str, meta_path: str) -> None:
        """Load FAISS index and metadata pickle from disk."""
        import faiss
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.index_path = str(index_path)
        self.meta_path = str(meta_path)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.is_ready():
            return []

        # Ensure float32 2D array
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)

        k = min(top_k, self.count())
        if k <= 0:
            return []

        distances, indices = self.index.search(vec, k)
        similarities = 1.0 / (1.0 + distances[0])

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, dist, sim in zip(indices[0], distances[0], similarities):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["_faiss_dist"] = float(dist)
            meta["_score"] = float(sim)
            results.append((meta, float(sim)))

        return results

    def add(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        import faiss
        vecs = np.asarray(vectors, dtype=np.float32)
        if self.index is None:
            self.dim = vecs.shape[1]
            self.index = faiss.IndexFlatL2(self.dim)

        self.index.add(vecs)
        self.metadata.extend(metadata)

    def count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0


class HNSWVectorStore(BaseVectorStore):
    """
    HNSW (hnswlib) vector store adapter.
    """

    def __init__(
        self,
        dim: int = 384,
        space: str = "cosine",
        max_elements: int = 100000
    ):
        self.dim = dim
        self.space = space
        self.max_elements = max_elements
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def _init_index(self):
        try:
            import hnswlib
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
        except ImportError:
            raise ImportError(
                "hnswlib is not installed. Run `pip install hnswlib` to use HNSWVectorStore."
            )

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if self.index is None:
            self._init_index()
        vecs = np.asarray(vectors, dtype=np.float32)
        start_idx = len(self.metadata)
        ids = np.arange(start_idx, start_idx + len(metadata))
        self.index.add_items(vecs, ids)
        self.metadata.extend(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.is_ready():
            return []
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec[0]
        k = min(top_k, self.count())
        labels, distances = self.index.knn_query(vec, k=k)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            sim = float(1.0 - dist) if self.space == "cosine" else float(1.0 / (1.0 + dist))
            meta = dict(self.metadata[idx])
            meta["_score"] = sim
            results.append((meta, sim))
        return results

    def count(self) -> int:
        return len(self.metadata)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store adapter.
    """

    def __init__(
        self,
        collection_name: str = "cybersec_rag",
        dim: int = 384,
        location: str = ":memory:"
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.location = location
        self.client = None
        self._count = 0

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            self.client = QdrantClient(location=self.location)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
        except ImportError:
            raise ImportError(
                "qdrant-client is not installed. Run `pip install qdrant-client` to use QdrantVectorStore."
            )

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if self.client is None:
            self._init_client()
        from qdrant_client.models import PointStruct
        vecs = np.asarray(vectors, dtype=np.float32)
        points = []
        for i, (v, meta) in enumerate(zip(vecs, metadata)):
            point_id = self._count + i
            points.append(PointStruct(id=point_id, vector=v.tolist(), payload=meta))
        self.client.upsert(collection_name=self.collection_name, points=points)
        self._count += len(points)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.is_ready():
            return []
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec[0]
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vec.tolist(),
            limit=top_k
        )
        results = []
        for hit in hits:
            meta = dict(hit.payload or {})
            score = float(hit.score)
            meta["_score"] = score
            results.append((meta, score))
        return results

    def count(self) -> int:
        return self._count

    def is_ready(self) -> bool:
        return self.client is not None and self._count > 0


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store adapter.
    """

    def __init__(
        self,
        collection_name: str = "cybersec_rag",
        persist_dir: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None

    def _init_client(self):
        try:
            import chromadb
            if self.persist_dir:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run `pip install chromadb` to use ChromaVectorStore."
            )

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if self.collection is None:
            self._init_client()
        vecs = np.asarray(vectors, dtype=np.float32)
        start_idx = self.collection.count()
        ids = [f"doc_{start_idx + i}" for i in range(len(metadata))]
        documents = [m.get("text", "") for m in metadata]
        self.collection.add(
            ids=ids,
            embeddings=vecs.tolist(),
            metadatas=metadata,
            documents=documents
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.is_ready():
            return []
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec[0]
        results = self.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=top_k
        )
        out = []
        if results and "metadatas" in results and results["metadatas"]:
            metas = results["metadatas"][0]
            dists = results.get("distances", [[0.0] * len(metas)])[0]
            for m, d in zip(metas, dists):
                sim = float(1.0 / (1.0 + d))
                meta_dict = dict(m)
                meta_dict["_score"] = sim
                out.append((meta_dict, sim))
        return out

    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    def is_ready(self) -> bool:
        return self.collection is not None and self.count() > 0


def create_vector_store(
    backend: Optional[str] = None,
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    **kwargs
) -> BaseVectorStore:
    """
    Factory function for instantiating vector store adapters.
    Selection is driven by `config.VECTOR_STORE_BACKEND` or runtime override.
    """
    selected = (backend or VECTOR_STORE_BACKEND or "faiss").lower().strip()

    if selected == "faiss":
        return FAISSVectorStore(index_path=index_path, meta_path=meta_path, **kwargs)
    elif selected == "hnsw":
        return HNSWVectorStore(**kwargs)
    elif selected == "qdrant":
        return QdrantVectorStore(**kwargs)
    elif selected == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unsupported vector store backend: '{selected}'. "
            f"Choose from: 'faiss', 'hnsw', 'qdrant', 'chroma'."
        )
