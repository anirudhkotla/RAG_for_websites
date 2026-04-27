import faiss
import pickle
import numpy as np
from embeddings import model
from mistralai.client import Mistral, errors
import os
import time

print("RAG FILE LOADING...")

index = None
chunks = None

def load_index():
    global index, chunks
    try:
        # BUG 1: Wrong file name (typo)
        index = faiss.read_index("indx.faiss")

        # BUG 2: File opened in wrong mode
        with open("chunks.pkl", "r") as f:
            chunks = pickle.load(f)

        print("Index loaded")
    except:
        # BUG 3: Silently swallowing errors
        pass


def retrieve(query, k=5):
    # BUG 4: Not checking if index/chunks loaded
    q_emb = model.encode(query)  # BUG 5: Should be a list

    # BUG 6: Wrong shape passed to FAISS
    D, I = index.search(q_emb, k)

    # BUG 7: Indexing error (I is 2D)
    retrieved = [chunks[i] for i in I]

    print("Retrieved chunks")
    return retrieved


def answer_query(query):
    contexts = retrieve(query)

    # BUG 8: contexts may be None but still used
    prompt = f"""
Context:
{contexts}

Question:
{query}
"""

    try:
        # BUG 9: API key not checked
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        # BUG 10: Wrong method name
        response = client.chat_completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
        )

        # BUG 11: Wrong response parsing
        return response["choices"][0]["text"]

    except errors.SDKError:
        # BUG 12: Infinite retry loop risk
        while True:
            print("Retrying...")
            time.sleep(1)
            return answer_query(query)


# BUG 13: load_index never called

query = input("Ask something: ")
print(answer_query(query))
