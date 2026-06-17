import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os

INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "documents.pkl"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "Employees must request leave 3 days in advance.",
    "VPN access requires manager approval.",
    "Working hours are 10 AM to 6 PM."
]

if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    print("Loading existing FAISS index and documents from disk...")

    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
else:
    print("No existing data. Creating new index from docs...")

    document_embeddings = embedding_model.encode(docs)
    dimention = document_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimention)
    index.add(np.array(document_embeddings).astype('float32'))

    faiss.write_index(index, INDEX_FILE)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    
    print("Embeddings computed and saved to disk.")


#query
query = "what is the working hour"
query_vector = embedding_model.encode([query]).astype('float32')

distance, indices = index.search(query_vector, k = 1)
retrived_doc = docs[indices[0][0]]

print("Retrived document: ", retrived_doc)

prompt = f"""
You are an enterprise assistant.

Use the context below to answer the question.

Context: {retrived_doc}
Question: {query}

"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    },
    timeout=60
)

print("\nLLM Answer:\n")
print(response.json()["response"])