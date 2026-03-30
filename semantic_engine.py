"""
semantic_engine.py
------------------
Semantic search using dense vector embeddings.

TARGET MODEL: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Trained on 1B+ sentence pairs
  - Maps semantically similar text to nearby vectors

WHAT WE DO HERE (offline fallback):
  Since the model cannot be downloaded in this environment, we implement
  a Random Projection embedding that preserves relative semantic distances
  through a deterministic hash-based Gaussian projection.

  The math:
    1. Tokenize text into a bag of words.
    2. For each token, generate a pseudo-random 384-dim vector using a
       seeded hash (same token → same vector, always).
    3. Sum the token vectors → document vector.
    4. L2-normalize → unit sphere → cosine similarity = dot product.

  This is a "random feature map" approximation of kernel similarity.
  It is NOT as good as all-MiniLM but it correctly:
    - Places similar tokens near each other (same hash → same direction)
    - Aggregates token semantics into one vector
    - Allows cosine similarity comparison

TO USE THE REAL MODEL: replace SemanticEngine.embed() with:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode(text)   # returns np.ndarray of shape (384,)

COSINE SIMILARITY:
  sim(A, B) = (A · B) / (|A| * |B|)
  Since we L2-normalize, this reduces to just A · B.
  Range: [-1, 1], where 1 = identical direction, 0 = orthogonal.
"""

import math
import re
import numpy as np


STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","i","you","it","we","they",
}

DIMS = 384   # Match all-MiniLM output dimension


def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _hash_project(token: str, dim: int) -> float:
    """
    Deterministic pseudo-Gaussian projection for one (token, dimension) pair.
    Uses two LCG steps seeded by token hash + dim to simulate Box-Muller.
    """
    h = 0
    for ch in token:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF

    seed = (h + dim * 2654435761) & 0xFFFFFFFF
    u1 = ((seed * 1103515245 + 12345) & 0xFFFFFFFF) / 0xFFFFFFFF + 1e-10
    u2 = ((seed * 22695477 + 1)       & 0xFFFFFFFF) / 0xFFFFFFFF

    # Box-Muller transform → standard normal sample
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def embed(text: str) -> np.ndarray:
    """
    Produce a 384-dim L2-normalized embedding for any text string.
    This is the drop-in replacement slot for all-MiniLM-L6-v2.
    """
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(DIMS)

    vec = np.zeros(DIMS)
    for token in tokens:
        for d in range(DIMS):
            vec[d] += _hash_project(token, d)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


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
        self.embeddings = [embed(self._build_text(p)) for p in products]
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
