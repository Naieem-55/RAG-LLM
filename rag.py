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
query = input("Question: ")
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

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

try:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}. Is it running? Try: ollama serve")
except requests.exceptions.HTTPError as e:
    raise SystemExit(f"Ollama error: {e}. Pull the model first: ollama pull {OLLAMA_MODEL}")
except requests.exceptions.Timeout:
    raise SystemExit("Ollama request timed out after 120s.")

print("\nQuestion: ", query)
print("Answer: ", response.json()["response"])