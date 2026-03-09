import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# === COnfig ===
Corpus_Path = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\corpus"
Index_path = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
Meta_path = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"
Model_name = "sentence-transformers/all-MiniLM-L6-v2"

# === Load Model ===

model = SentenceTransformer(Model_name)

# === Load File ===

documents = []
metadata = []

for filename in os.listdir(Corpus_Path):
    if filename.endswith(".txt"):
        with open(os.path.join(Corpus_Path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            metadata.append({"filename": filename})

# === Generate Embeddings ===

embeddings = model.encode(documents, show_progress_bar=True)

# === Build FAISS Index ===

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Save Index and Metadata ===

faiss.write_index(index, Index_path)

with open(Meta_path, "wb") as f:
    pickle.dump(metadata, f)

print("index built and saved successfully")