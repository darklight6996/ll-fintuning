# ==========================================
# rag_pipeline_v2.py
# Production Style CPU RAG Pipeline
# ==========================================

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import textwrap

# ==========================================
# PATHS
# ==========================================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
META_PATH  = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

# ==========================================
# SETTINGS
# ==========================================

TOP_K = 10          # initial FAISS retrieval
FINAL_K = 3         # reranked final docs
MAX_CONTEXT_CHARS = 2500

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading LLM...")
MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")

# ==========================================
# LOAD INDEX + METADATA
# ==========================================

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("\n✅ RAG Pipeline Ready\n")

# ==========================================
# HELPERS
# ==========================================

def build_context(top_docs):
    """
    Prevent too much context from overflowing model limits
    """
    context = ""
    used_sources = []

    for doc in top_docs:
        text = doc["text"]
        filename = doc["filename"]

        chunk = f"[SOURCE: {filename}]\n{text}\n\n"

        if len(context) + len(chunk) <= MAX_CONTEXT_CHARS:
            context += chunk
            used_sources.append(filename)
        else:
            break

    return context, used_sources


def ask_llm(context, query):
    prompt = f"""
You are a cybersecurity technical assistant.

Use ONLY the provided context to answer.

Rules:
1. Be clear and technical.
2. Give step-by-step explanation when needed.
3. If answer is not found, say:
Not found in provided context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
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
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    query = input("Enter your query (or exit): ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    # ======================================
    # STEP 1: EMBED QUERY
    # ======================================

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # ======================================
    # STEP 2: FAISS SEARCH
    # ======================================

    distances, indices = index.search(query_embedding, TOP_K)

    candidates = []
    for idx in indices[0]:
        if idx != -1:
            candidates.append(metadata[idx])

    # ======================================
    # STEP 3: RERANK RESULTS
    # ======================================

    pairs = [[query, doc["text"]] for doc in candidates]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, score in ranked[:FINAL_K]]

    # ======================================
    # STEP 4: BUILD SAFE CONTEXT
    # ======================================

    context, sources = build_context(top_docs)

    # ======================================
    # STEP 5: GENERATE ANSWER
    # ======================================

    answer = ask_llm(context, query)

    # ======================================
    # STEP 6: PRINT OUTPUT
    # ======================================

    print("\n" + "="*70)
    print("ANSWER:\n")

    wrapped = textwrap.fill(answer, width=100)
    print(wrapped)

    print("\nSOURCES USED:")
    for s in sources:
        print("-", s)

    print("="*70 + "\n")