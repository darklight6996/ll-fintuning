# ==========================================
# rag_pipeline_v3_gpu.py
# RTX 3060 Ti 8GB Optimized GPU RAG Pipeline
# ==========================================

import faiss
import pickle
import numpy as np
import torch
import textwrap

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ==========================================
# PATHS
# ==========================================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
META_PATH  = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

# ==========================================
# SETTINGS
# ==========================================

TOP_K = 10
FINAL_K = 3
MAX_CONTEXT_CHARS = 3200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

print("Loading reranker...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=DEVICE
)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading GPU LLM...")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# 4-bit quantization for 8GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n✅ GPU Pipeline Ready\n")

# ==========================================
# HELPERS
# ==========================================

def build_context(top_docs):
    context = ""
    sources = []

    for doc in top_docs:
        filename = doc["filename"]
        text = doc["text"]

        block = f"[SOURCE: {filename}]\n{text}\n\n"

        if len(context) + len(block) <= MAX_CONTEXT_CHARS:
            context += block
            sources.append(filename)
        else:
            break

    return context, sources


def generate_answer(query, context):
    prompt = f"""
You are a cybersecurity technical assistant.

Use ONLY the supplied context.

Rules:
1. Be accurate and technical.
2. Give step-by-step explanations when relevant.
3. If missing from context, say:
Not found in provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()

    return decoded


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    query = input("Enter your query (or exit): ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    # ===============================
    # STEP 1: EMBEDDING
    # ===============================

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # ===============================
    # STEP 2: FAISS SEARCH
    # ===============================

    distances, indices = index.search(query_embedding, TOP_K)

    candidates = []
    for idx in indices[0]:
        if idx != -1:
            candidates.append(metadata[idx])

    # ===============================
    # STEP 3: RERANK
    # ===============================

    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, _ in ranked[:FINAL_K]]

    # ===============================
    # STEP 4: CONTEXT BUILD
    # ===============================

    context, sources = build_context(top_docs)

    # ===============================
    # STEP 5: GENERATE
    # ===============================

    answer = generate_answer(query, context)

    # ===============================
    # STEP 6: OUTPUT
    # ===============================

    print("\n" + "=" * 70)
    print("ANSWER:\n")
    print(textwrap.fill(answer, width=100))

    print("\nSOURCES USED:")
    for s in sources:
        print("-", s)

    print("=" * 70 + "\n")