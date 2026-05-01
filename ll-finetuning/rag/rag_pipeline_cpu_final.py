# ==========================================================
# rag_pipeline_final_cpu.py
# FINAL CPU RAG PIPELINE
# ==========================================================

import faiss
import pickle
import json
import os
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# PATHS
# ==========================================================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"

META_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

MEMORY_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\memory.json"

# ==========================================================
# SETTINGS
# ==========================================================

TOP_K = 10
FINAL_K = 3
MAX_CONTEXT_CHARS = 3500
MAX_MEMORY_TURNS = 6

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading reranker...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading LLM...")
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")

# ==========================================================
# MEMORY
# ==========================================================

def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

memory = load_memory()

# ==========================================================
# HELPERS
# ==========================================================

def get_recent_memory():
    recent = memory[-MAX_MEMORY_TURNS:]
    text = ""

    for item in recent:
        text += f"User: {item['user']}\n"
        text += f"Assistant: {item['assistant']}\n\n"

    return text.strip()

def build_context(top_docs):
    context = ""

    for doc in top_docs:
        block = (
            f"[SOURCE: {doc['filename']} "
            f"Chunk {doc['chunk_id']}]\n"
            f"{doc['text']}\n\n"
        )

        if len(context) + len(block) <= MAX_CONTEXT_CHARS:
            context += block
        else:
            break

    return context.strip()

def retrieve_docs(query):

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    candidates = [
        metadata[i]
        for i in indices[0]
        if i != -1
    ]

    pairs = [
        [query, doc["text"]]
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:FINAL_K]]

def ask_llm(query, context, memory_text):

    prompt = f"""
Use the memory and context to answer clearly.

Conversation Memory:
{memory_text}

Knowledge Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

# ==========================================================
# MAIN LOOP
# ==========================================================

print("\n✅ FINAL CPU RAG READY\n")

while True:

    query = input(
        "Enter query (exit/reset/history/forget): "
    ).strip()

    if not query:
        continue

    if query.lower() == "exit":
        break

    if query.lower() == "reset":
        memory = []
        save_memory(memory)
        print("Memory cleared.\n")
        continue

    if query.lower() == "history":
        for i, item in enumerate(memory, 1):
            print(f"{i}. {item['user']}")
        print()
        continue

    if query.lower() == "forget":
        if memory:
            memory.pop()
            save_memory(memory)
            print("Last memory removed.\n")
        continue

    top_docs = retrieve_docs(query)
    context = build_context(top_docs)
    memory_text = get_recent_memory()

    answer = ask_llm(
        query,
        context,
        memory_text
    )

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")
    for doc in top_docs:
        print(
            f"- {doc['filename']} "
            f"(Chunk {doc['chunk_id']})"
        )

    print("\n" + "=" * 60 + "\n")

    memory.append({
        "user": query,
        "assistant": answer
    })

    save_memory(memory)