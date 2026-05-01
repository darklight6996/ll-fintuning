# ==========================================================
# rag_pipeline_v4_memory_gpu.py
# RAG + Memory + ReRanker + GPU (RTX 3060 Ti 8GB)
# ==========================================================

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

# ==========================================================
# PATHS
# ==========================================================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
META_PATH  = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

# ==========================================================
# SETTINGS
# ==========================================================

TOP_K = 10
FINAL_K = 3
MAX_HISTORY = 4
MAX_CONTEXT_CHARS = 3500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# LOAD MODELS
# ==========================================================

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

print("Loading LLM...")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

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

print("\n✅ v4 Memory GPU Pipeline Ready\n")

# ==========================================================
# MEMORY
# ==========================================================

chat_history = []

# ==========================================================
# HELPERS
# ==========================================================

def get_recent_history():
    recent = chat_history[-MAX_HISTORY:]
    history_text = ""

    for turn in recent:
        history_text += f"User: {turn['user']}\n"
        history_text += f"Assistant: {turn['assistant']}\n\n"

    return history_text.strip()


def build_context(top_docs):
    context = ""

    for doc in top_docs:
        block = f"[SOURCE: {doc['filename']}]\n{doc['text']}\n\n"

        if len(context) + len(block) <= MAX_CONTEXT_CHARS:
            context += block
        else:
            break

    return context.strip()


def generate_answer(query, history, context):

    prompt = f"""
You are a cybersecurity technical assistant.

Use the previous conversation for continuity.
Use the supplied context for facts.

If answer is not in context, say:
Not found in provided context.

Previous Conversation:
{history}

Context:
{context}

Current Question:
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


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    query = input("Enter query (exit/reset/history): ").strip()

    # EXIT
    if query.lower() == "exit":
        break

    # RESET MEMORY
    if query.lower() == "reset":
        chat_history = []
        print("✅ Memory cleared\n")
        continue

    # SHOW MEMORY
    if query.lower() == "history":
        print("\n===== CHAT HISTORY =====")
        if len(chat_history) == 0:
            print("No memory stored.")
        else:
            for i, turn in enumerate(chat_history, 1):
                print(f"{i}. User: {turn['user']}")
                print(f"   Assistant: {turn['assistant'][:200]}")
        print("========================\n")
        continue

    if not query:
        continue

    # ======================================================
    # STEP 1: EMBEDDING
    # ======================================================

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # ======================================================
    # STEP 2: SEARCH
    # ======================================================

    distances, indices = index.search(query_embedding, TOP_K)

    candidates = []
    for idx in indices[0]:
        if idx != -1:
            candidates.append(metadata[idx])

    # ======================================================
    # STEP 3: RERANK
    # ======================================================

    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, _ in ranked[:FINAL_K]]

    # ======================================================
    # STEP 4: BUILD HISTORY + CONTEXT
    # ======================================================

    history = get_recent_history()
    context = build_context(top_docs)

    # ======================================================
    # STEP 5: GENERATE
    # ======================================================

    answer = generate_answer(query, history, context)

    # ======================================================
    # STEP 6: SAVE MEMORY
    # ======================================================

    chat_history.append({
        "user": query,
        "assistant": answer
    })

    # ======================================================
    # STEP 7: OUTPUT
    # ======================================================

    print("\n" + "=" * 70)
    print("ANSWER:\n")
    print(textwrap.fill(answer, width=100))
    print("=" * 70 + "\n")