"""
bm25_engine.py
--------------
BM25 (Best Match 25) retrieval — implemented from scratch without libraries.

HOW BM25 WORKS:
  BM25 is a probabilistic ranking function, an evolution of TF-IDF.
  For each query term t, and document d:

      IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

      TF_norm(t, d) = f(t,d) * (k1 + 1)
                      ─────────────────────────────────
                      f(t,d) + k1 * (1 - b + b * |d|/avgdl)

      Score(d, Q) = Σ_t  IDF(t) * TF_norm(t, d)

  Parameters:
    k1 = 1.5  — controls TF saturation (higher = slower saturation)
    b  = 0.75 — controls document length normalization (1 = full, 0 = none)

WHY BM25 BEATS PLAIN TF-IDF:
  1. Length normalization: short product titles aren't unfairly penalised.
  2. TF saturation: the 50th occurrence of "black" adds almost nothing.
  3. IDF formula is smoother and handles edge cases better.
"""

import math
import re
from collections import Counter


STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","i","you","it","we","they",
    "this","that","these","those","what","how","im","need","want","get","buy",
}


def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Engine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.products = []
        self.corpus   = []   # list of token lists
        self.idf      = {}   # term -> IDF score
        self.avgdl    = 0.0

    def fit(self, products: list[dict], text_fn) -> None:
        self.products = products
        self.corpus   = [tokenize(text_fn(p)) for p in products]
        N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / N

        # Document frequency per term
        df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # BM25 IDF (Robertson-Sparck Jones variant, smooth)
        self.idf = {
            t: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for t, freq in df.items()
        }
        print(f"[BM25] Fitted. Corpus size: {N}, Avg doc length: {self.avgdl:.1f} tokens")

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        doc   = self.corpus[doc_index]
        dl    = len(doc)
        tf    = Counter(doc)
        score = 0.0

        for term in query_tokens:
            if term not in self.idf:
                continue
            f = tf.get(term, 0)
            numerator   = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * (numerator / denominator)

        return score

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top_k products ranked by BM25 score.
        Each result: { product, score, rank }
        """
        q_tokens = tokenize(query_text)

        scores = [
            (i, self._score(q_tokens, i))
            for i in range(len(self.products))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k]):
            results.append({
                "product": self.products[idx],
                "score":   score,
                "rank":    rank + 1,
            })
        return results
