"""
core/query_index.py

Interactive query script for testing FAISS document vector index retrieval.
"""

import os
import sys
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.config import INDEX_PATH, META_PATH, EMBED_MODEL


def main():
    print("=" * 60)
    print("FAISS Document Index Query Utility")
    print("=" * 60)

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print(f"[ERROR] Index or metadata file missing.")
        print(f"Index Path: {INDEX_PATH}")
        print(f"Meta Path:  {META_PATH}")
        print("Please run `python core/build_index.py` first.")
        return

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Loading FAISS Index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading Metadata...")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded index with {len(metadata)} documents.\n")

    while True:
        try:
            query = input("Enter query (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        query_embedding = model.encode([query])
        k = min(3, len(metadata))

        distances, indices = index.search(query_embedding, k)

        print("\nTop Results:\n")
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            doc = metadata[idx]
            print(f"Result {i+1}:")
            print(f"  Filename: {doc.get('filename', 'N/A')}")
            print(f"  Distance: {distances[0][i]:.4f}")
            print(f"  Snippet:  {doc.get('text', '')[:200]}...")
            print("-" * 50)
        print()


if __name__ == "__main__":
    main()
