# ==========================================================
# core/benchmark.py
# QUICK QUALITY TESTER FOR YOUR RAG SYSTEM
# Measures retrieval relevance + answer speed
# ==========================================================

import os
import sys
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.config import (
    INDEX_PATH,
    META_PATH,
    TOP_K,
    FINAL_K,
    EMBED_MODEL,
    RERANK_MODEL
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Loading FAISS...")
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    print(f"[ERROR] Index or metadata missing. Run build_index.py first.")
    sys.exit(1)

index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# ==========================================================
# TEST QUERIES
# ==========================================================

tests = [
    "what is port 443",
    "how does privilege escalation work",
    "what is sql injection",
    "what is kerberos",
    "how to detect brute force attempts"
]

# ==========================================================
# RETRIEVAL FUNCTION
# ==========================================================

def search(query):
    q = embed_model.encode([query])
    q = np.array(q, dtype=np.float32)

    distances, indices = index.search(q, TOP_K)

    candidates = [metadata[i] for i in indices[0] if i != -1 and i < len(metadata)]

    if not candidates:
        return []

    pairs = [[query, doc.get("text", "")] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:FINAL_K]

# ==========================================================
# RUN TESTS
# ==========================================================

def main():
    print("\n===== BENCHMARK START =====\n")

    for query in tests:
        start = time.time()
        results = search(query)
        end = time.time()

        print("=" * 70)
        print("QUERY:", query)
        print(f"TIME : {end-start:.2f}s\n")

        for rank, (doc, score) in enumerate(results, 1):
            print(f"Rank {rank} | Score {score:.4f}")
            print(f"File: {doc.get('filename', 'N/A')} | Chunk: {doc.get('chunk_id', 'N/A')}")
            print(doc.get("text", "")[:350])
            print("-" * 70)

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()