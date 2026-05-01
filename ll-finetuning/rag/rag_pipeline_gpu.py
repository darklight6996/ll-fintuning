import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ==============================
# DEVICE
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# PATHS
# ==============================

INDEX_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
META_PATH  = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

TOP_K = 10
FINAL_K = 3

# ==============================
# LOAD MODELS
# ==============================

print("Loading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Loading reranker...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading LLM...")
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)

print(f"\n✅ GPU Pipeline ready on {device}\n")

# ==============================
# LOOP
# ==============================

while True:
    query = input("Enter your query (or 'exit'): ").strip()

    if query.lower() == "exit":
        break
    if not query:
        continue

    # Embed query
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # FAISS retrieval
    distances, indices = index.search(query_embedding, TOP_K)

    candidates = [
        metadata[idx]["text"]
        for idx in indices[0] if idx != -1
    ]

    # Re-ranking
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, _ in ranked[:FINAL_K]]
    context = "\n\n".join(top_docs)

    # Prompt
    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question: {query}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nAnswer:\n", answer)
    print("\n" + "="*60 + "\n")