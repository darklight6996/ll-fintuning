# ==========================================================
# benchmark_rag_final.py
# QUICK QUALITY TESTER FOR YOUR RAG SYSTEM
# Measures retrieval relevance + answer speed
# ==========================================================

import time
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================================
# PATHS
# ==========================================================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
META_PATH  = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

TOP_K = 10
FINAL_K = 3

# ==========================================================
# LOAD
# ==========================================================

print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading FAISS...")
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

    candidates = [metadata[i] for i in indices[0] if i != -1]

    pairs = [[query, doc["text"]] for doc in candidates]
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
        print(f"File: {doc['filename']} | Chunk: {doc['chunk_id']}")
        print(doc["text"][:350])
        print("-" * 70)

print("\n===== DONE =====")