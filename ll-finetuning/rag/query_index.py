import faiss
import pickle
from sentence_transformers import SentenceTransformer

# paths

Index_path = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\faiss.index"
Meta_path = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\rag\faiss_index\meta.pkl"

# Load Embedding Model

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS Index 

index = faiss.read_index(Index_path)

# Load Metadata

with open(Meta_path, "rb") as f:
    metadata = pickle.load(f)

print("Index and metadata loaded successfully")

while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break


    # convert query to embedding 

    query_embedding = model.encode([query])

    # search vector database

    k = 3 # number of nearest neighbors to retrieve

    distances, indices = index.search(query_embedding, k)

    print("\nTop results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Result {i+1}:")
        print(f"Filename: {metadata[idx]['filename']}")
        print(f"Distance: {distances[0][i]:.4f}\n")
        print("-" * 50)


