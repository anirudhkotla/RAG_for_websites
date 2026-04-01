import faiss
import pickle
import numpy as np
from embeddings import model
from mistralai.client import Mistral, errors
import os
import time

print("RAG FILE LOADING...")

def load_index():
    try:
        index = faiss.read_index("index.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        print(f"FAISS index loaded with {len(chunks)} chunks")
        return index, chunks
    except Exception as e:
        print(f"Failed to load FAISS index or chunks: {e}")
        return None, None

def retrieve(query, k=5):
    index, chunks = load_index()
    if index is None or chunks is None:
        print("Index or chunks missing, cannot retrieve context")
        return []

    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    retrieved = [chunks[i] for i in I[0]]
    print(f"Retrieved {len(retrieved)} chunks for query: {query}")
    return retrieved

def answer_query(query, max_retries=3, backoff=2):
    """
    Sends a query to Mistral chat API with retry logic.
    """
    contexts = retrieve(query)
    if not contexts:
        return "No context available to answer your query."

    print("Mistral API ready")
    prompt = f"""
Answer ONLY from the context below.

Context:
{contexts}

Question:
{query}
"""

    attempt = 0
    while attempt < max_retries:
        try:
            with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as client:
                response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                )
            return response.choices[0].message.content

        except errors.SDKError as e:
            if hasattr(e, "http_res") and str(e.http_res.status_code).startswith("5"):
                attempt += 1
                wait_time = backoff ** attempt
                print(f"Transient error ({e.http_res.status_code}). Retry {attempt}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Non-retryable API error: {e}")
                raise e

    return "Sorry, the service is temporarily unavailable. Please try again later."