"""
core/build_index.py

Builds FAISS vector index and metadata pickle file from corpus text documents.
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

from core.config import (
    CORPUS_DIR,
    INDEX_PATH,
    META_PATH,
    FAISS_INDEX_DIR,
    EMBED_MODEL
)


def build():
    print("=" * 60)
    print("Building Document FAISS Index")
    print(f"Corpus Directory: {CORPUS_DIR}")
    print(f"Index Target:     {INDEX_PATH}")
    print("=" * 60)

    if not os.path.exists(CORPUS_DIR):
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    print(f"\nLoading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    documents = []
    metadata = []

    print("\nReading corpus documents...")
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(CORPUS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        documents.append(text)
                        metadata.append({
                            "filename": filename,
                            "text": text,
                            "chunk_id": os.path.splitext(filename)[0]
                        })
            except Exception as e:
                print(f"[WARNING] Could not read {filename}: {e}")

    if not documents:
        print("[WARNING] No documents found in corpus directory.")
        return

    print(f"Loaded {len(documents)} document(s).")
    print("Generating embeddings...")

    embeddings = model.encode(documents, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("\nIndex built and saved successfully.")
    print(f"Saved FAISS Index: {INDEX_PATH}")
    print(f"Saved Metadata:    {META_PATH}")


if __name__ == "__main__":
    build()