"""
semantic_engine.py
------------------
Semantic search using dense vector embeddings with all-MiniLM-L6-v2.

MODEL: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Trained on 1B+ sentence pairs
  - Maps semantically similar text to nearby vectors
  - Lightweight and efficient for both CPU and GPU inference

COSINE SIMILARITY:
  sim(A, B) = (A · B) / (|A| * |B|)
  Range: [-1, 1], where 1 = identical direction, 0 = orthogonal.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Initialize the all-MiniLM model
print("[Semantic] Loading all-MiniLM-L6-v2 model...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[Semantic] Model loaded successfully.")
DIMS = 384   # all-MiniLM-L6-v2 output dimension


def embed(text: str) -> np.ndarray:
    """
    Produce a 384-dim L2-normalized embedding using all-MiniLM-L6-v2.
    
    Args:
        text: The text to embed.
        
    Returns:
        A 384-dimensional numpy array representing the semantic embedding.
    """
    if not text or not text.strip():
        return np.zeros(DIMS)
    
    embedding = MODEL.encode(text, convert_to_numpy=True, show_progress=False)
    return embedding


class SemanticEngine:
    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.products: list[dict] = []

    def _build_text(self, p: dict) -> str:
        """
        Richer text for embedding: add price-range semantics so queries like
        'affordable' or 'budget' can match cheap products.
        """
        if p["price"] < 200:
            price_label = "budget cheap affordable"
        elif p["price"] < 600:
            price_label = "mid range"
        else:
            price_label = "premium high end flagship professional"

        return (
            f"{p['title']} {p['title']} "
            f"{p['description']} "
            f"{p['manufacturer']} {p['category']} {p['color']} "
            f"{price_label}"
        )

    def fit(self, products: list[dict]) -> None:
        self.products = products
        # Build text for all products, then batch encode for efficiency
        texts = [self._build_text(p) for p in products]
        self.embeddings = MODEL.encode(texts, convert_to_numpy=True, show_progress=False)
        print(f"[Semantic] Fitted. {len(products)} embeddings of {DIMS} dims each.")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top_k products by cosine similarity to query embedding.
        Each result: { product, score, rank }
        """
        q_vec = embed(query_text)

        # Dot product with each document vector (both are L2-normalized)
        scores = [(i, float(np.dot(q_vec, doc_vec)))
                  for i, doc_vec in enumerate(self.embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k]):
            results.append({
                "product": self.products[idx],
                "score":   score,
                "rank":    rank + 1,
            })
        return results
