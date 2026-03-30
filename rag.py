"""
rag.py
------
Main RAG (Retrieval-Augmented Generation) pipeline.

FULL PIPELINE:
  1. [database.py]       Load products from SQLite
  2. [tfidf_engine.py]   Fit TF-IDF matrix on corpus
  3. [bm25_engine.py]    Fit BM25 index on corpus
  4. [semantic_engine.py] Compute 384-dim embeddings for all products
  5. [query_parser.py]   Parse user query → structured JSON
  6. Run all 3 retrievers on the augmented query
  7. [fusion.py]         Merge ranked lists via Reciprocal Rank Fusion
  8. Post-filter by budget if provided
  9. [fusion.py]         Display similarity table

Usage:
    python rag.py
    python rag.py "black gaming laptop under 1300"
"""

import sys
from database      import load_products, product_to_text, create_db
from tfidf_engine  import TFIDFEngine
from bm25_engine   import BM25Engine
from semantic_engine import SemanticEngine
from query_parser  import parse_query
from fusion        import reciprocal_rank_fusion, show_similarities


# ── One-time setup ─────────────────────────────────────────────────────────────

def build_rag():
    """Load DB and fit all three retrieval engines. Returns the RAG object."""
    products = load_products()

    tfidf    = TFIDFEngine()
    bm25     = BM25Engine()
    semantic = SemanticEngine()

    tfidf.fit(products, product_to_text)
    bm25.fit(products, product_to_text)
    semantic.fit(products)

    return {"tfidf": tfidf, "bm25": bm25, "semantic": semantic, "products": products}


# ── Search ─────────────────────────────────────────────────────────────────────

def search(rag: dict, query: str, top_k: int = 5) -> list[dict]:
    """
    Run the full RAG pipeline for a single query.

    Steps:
      1. Parse query into structured fields
      2. Build augmented query string (original + category + color)
      3. Run TF-IDF, BM25, Semantic in parallel
      4. Apply budget filter to each result list
      5. Fuse ranked lists with RRF
      6. Display similarity table
      7. Print final results
    """
    print(f"\n{'─'*60}")
    print(f" QUERY: {query}")
    print(f"{'─'*60}")

    # Step 1 — parse
    parsed = parse_query(query)
    print(f" Parsed → category={parsed['category']}, color={parsed['color']}, "
          f"budget={parsed['budget']}, model={parsed['model']}")

    # Step 2 — augment query with extracted fields
    augmented = query
    if parsed["category"]: augmented += f" {parsed['category']}"
    if parsed["color"]:    augmented += f" {parsed['color']}"
    if parsed["model"]:    augmented += f" {parsed['model']}"

    # Step 3 — retrieve (each returns top 10 for fusion headroom)
    tf_res  = rag["tfidf"].query(augmented, top_k=10)
    bm_res  = rag["bm25"].query(augmented, top_k=10)
    sem_res = rag["semantic"].query(augmented, top_k=10)

    # Step 4 — budget filter
    budget = parsed["budget"]
    if budget:
        tf_res  = [r for r in tf_res  if r["product"]["price"] <= budget]
        bm_res  = [r for r in bm_res  if r["product"]["price"] <= budget]
        sem_res = [r for r in sem_res if r["product"]["price"] <= budget]
        print(f" Budget filter: ≤ ${budget:.0f}")

    # Step 5 — RRF fusion
    fused = reciprocal_rank_fusion([tf_res, bm_res, sem_res], top_k=top_k)

    # Step 6 — show similarity table
    show_similarities(query, tf_res, bm_res, sem_res, fused)

    # Step 7 — print final ranked results
    print(" FINAL TOP RESULTS (RRF Fusion):")
    print(f" {'#':<3} {'Product':<38} {'Manufacturer':<12} {'Price':>7}")
    print(f" {'─'*3} {'─'*38} {'─'*12} {'─'*7}")
    for r in fused:
        p = r["product"]
        print(f" {r['rank']:<3} {p['title']:<38} {p['manufacturer']:<12} ${p['price']:>6.0f}")

    return fused


# ── CLI entry point ─────────────────────────────────────────────────────────────

DEFAULT_QUERIES = [
    "black gaming laptop under 1300",
    "noise cancelling headphones for travel",
    "flagship smartphone with great camera",
    "4K TV for PS5 gaming under 1500",
    "affordable wireless mouse",
]

if __name__ == "__main__":
    create_db()           # create DB if it doesn't exist
    rag = build_rag()     # fit all engines once

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = DEFAULT_QUERIES

    for q in queries:
        search(rag, q)
