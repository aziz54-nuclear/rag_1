"""
tfidf_engine.py
---------------
TF-IDF retrieval using scikit-learn's TfidfVectorizer.

HOW TF-IDF WORKS:
  TF  (Term Frequency)  = how often a term appears in a document
  IDF (Inverse Doc Freq) = log(N / df(t))  — rare terms get higher weight
  Score = TF * IDF for each term, then cosine similarity between query and doc.

WHY IT WORKS FOR SEARCH:
  A query like "gaming laptop" will strongly match documents containing both
  words because "gaming" and "laptop" are not common across all documents,
  so their IDF is high.

LIMITATION:
  Purely lexical — "notebook" won't match "laptop" unless they share tokens.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams (e.g. "noise cancelling")
            min_df=1,
            sublinear_tf=True,    # replace TF with 1 + log(TF) to dampen high freq
            stop_words="english",
        )
        self.matrix = None    # shape: (n_docs, vocab_size)
        self.products = []

    def fit(self, products: list[dict], text_fn) -> None:
        """
        Build the TF-IDF matrix from the product corpus.
        text_fn: function that converts a product dict to a string.
        """
        self.products = products
        corpus = [text_fn(p) for p in products]
        self.matrix = self.vectorizer.fit_transform(corpus)
        print(f"[TF-IDF] Fitted. Vocab size: {len(self.vectorizer.vocabulary_)}")

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top_k products ranked by cosine similarity to query.
        Each result: { product, score, rank }
        """
        q_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        # Rank by descending score
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            results.append({
                "product": self.products[idx],
                "score":   float(scores[idx]),
                "rank":    rank + 1,
            })
        return results
