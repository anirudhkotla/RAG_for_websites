import faiss
import pickle
import numpy as np
import os
import time
from functools import lru_cache
from embeddings import model
from mistralai.client import Mistral, errors

class RAGSystem:
    def __init__(self, index_path="index.faiss", chunks_path="chunks.pkl"):
        print("Initializing RAG system...")
        self.index = None
        self.chunks = None
        self._load_resources(index_path, chunks_path)

        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set in environment variables")

    def _load_resources(self, index_path, chunks_path):
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"Loaded FAISS index with {len(self.chunks)} chunks")
        except Exception as e:
            print(f"[ERROR] Failed to load resources: {e}")

    @lru_cache(maxsize=128)
    def _embed_query(self, query):
        return model.encode([query])

    def retrieve(self, query, k=5):
        if self.index is None or self.chunks is None:
            print("[WARN] Index or chunks not loaded")
            return []

        try:
            q_emb = self._embed_query(query)
            D, I = self.index.search(np.array(q_emb), k)

            retrieved = []
            for idx in I[0]:
                if 0 <= idx < len(self.chunks):
                    retrieved.append(self.chunks[idx])

            print(f"Retrieved {len(retrieved)} chunks")
            return retrieved

        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return []

    def answer_query(self, query, max_retries=3, backoff=2):
        contexts = self.retrieve(query)

        if not contexts:
            return "No relevant context found."

        prompt = f"""
Answer ONLY using the provided context.

Context:
{contexts}

Question:
{query}
"""

        attempt = 0

        while attempt < max_retries:
            try:
                with Mistral(api_key=self.api_key) as client:
                    response = client.chat.complete(
                        model="mistral-large-latest",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,  # more deterministic
                    )
                return response.choices[0].message.content

            except errors.SDKError as e:
                status = getattr(e, "http_res", None)

                if status and str(status.status_code).startswith("5"):
                    attempt += 1
                    wait_time = backoff ** attempt
                    print(f"[Retry {attempt}] Server error. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Non-retryable error: {e}")
                    return "An error occurred while processing your request."

        return "Service temporarily unavailable. Try again later."


# ---- Usage ----
if __name__ == "__main__":
    rag = RAGSystem()

    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag.answer_query(query)
        print("\nAnswer:\n", answer)
