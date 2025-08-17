# src/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []

    def build_index(self, documents):
        """
        documents: list of strings
        """
        self.corpus = documents
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, text, top_k=3):
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index first.")

        q_emb = self.model.encode([text], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        results = [(self.corpus[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results

# quick test
if __name__ == "__main__":
    docs = ["Leaked DB dump from forum", "Normal chit chat", "Ransomware sample link"]
    retriever = Retriever()
    retriever.build_index(docs)
    print(retriever.query("database leak"))
