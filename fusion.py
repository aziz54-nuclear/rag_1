"""
fusion.py
---------
Reciprocal Rank Fusion (RRF) — combines ranked lists from multiple retrieval methods.

HOW RRF WORKS:
  Given K ranked lists R1, R2, ..., Rk (one per retrieval method), the RRF
  score for a document d is:

      RRF(d) = Σ_i  1 / (k + rank_i(d))

  where:
    k    = smoothing constant (default 60, from the original paper)
    rank_i(d) = position of document d in list i (1-indexed)

  If a document doesn't appear in a list, it contributes 0 for that list.

WHY RRF:
  - No need to normalize scores across methods (TF-IDF scores ≠ BM25 scores)
  - Robust: a document ranked #1 by one method and ignored by others still
    ranks high — it doesn't get buried by the other methods' zeros.
  - Simple and empirically strong (often beats learned fusion).

SIMILARITY DISPLAY:
  show_similarities() normalizes each method's raw scores to [0, 1] and
  prints a side-by-side comparison table so you can see where methods agree
  and where they diverge.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Merge ranked result lists from multiple retrieval methods using RRF.

    Args:
        ranked_lists: list of result lists, each from one engine.
                      Each result must have keys: 'product' (dict with 'id') and 'rank'.
        k:            RRF smoothing constant (60 per original paper).
        top_k:        How many final results to return.

    Returns:
        List of dicts: { product, rrf_score, rank }
    """
    rrf_scores: dict[int, float] = {}
    product_by_id: dict[int, dict] = {}

    for ranked_list in ranked_lists:
        for result in ranked_list:
            pid = result["product"]["id"]
            product_by_id[pid] = result["product"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + result["rank"])

    sorted_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)

    return [
        {
            "product":   product_by_id[pid],
            "rrf_score": rrf_scores[pid],
            "rank":      rank + 1,
        }
        for rank, pid in enumerate(sorted_ids[:top_k])
    ]


def normalize_scores(results: list[dict], score_key: str = "score") -> list[dict]:
    """
    Normalize raw scores to [0, 1] using min-max normalization.
    Returns a new list with an added 'normalized' key.
    """
    if not results:
        return results
    vals = [r[score_key] for r in results]
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    return [{**r, "normalized": (r[score_key] - lo) / rng} for r in results]


def show_similarities(
    query: str,
    tfidf_results: list[dict],
    bm25_results:  list[dict],
    sem_results:   list[dict],
    fusion_results: list[dict],
) -> None:
    """
    Print a formatted similarity comparison table.
    Rows = top products from fusion; columns = each retrieval method's score.
    """
    # Build lookup: product_id -> normalized score per method
    def build_map(results: list[dict], key: str = "score") -> dict[int, float]:
        normed = normalize_scores(results, key)
        return {r["product"]["id"]: r["normalized"] for r in normed}

    tf_map  = build_map(tfidf_results)
    bm_map  = build_map(bm25_results)
    sem_map = build_map(sem_results)

    # Normalize RRF scores too
    fusion_normed = normalize_scores(fusion_results, "rrf_score")
    fus_map = {r["product"]["id"]: r["normalized"] for r in fusion_normed}

    # Header
    print("\n" + "=" * 78)
    print(f" SIMILARITY RESULTS  |  Query: \"{query}\"")
    print("=" * 78)
    print(f"  {'#':<3} {'Product':<32} {'TF-IDF':>7} {'BM25':>7} {'Semantic':>9} {'Fusion':>7}")
    print("-" * 78)

    for r in fusion_results:
        pid   = r["product"]["id"]
        title = r["product"]["title"][:30]
        tf    = tf_map.get(pid, 0.0)
        bm    = bm_map.get(pid, 0.0)
        sem   = sem_map.get(pid, 0.0)
        fus   = fus_map.get(pid, 0.0)
        bar   = _bar(fus)
        print(
            f"  {r['rank']:<3} {title:<32} "
            f"{tf:>6.1%} {bm:>6.1%} {sem:>8.1%} {fus:>6.1%}  {bar}"
        )

    print("=" * 78)
    print(f"  Scores normalized to [0,1] within each method. "
          f"Fusion = RRF rank merging.")
    print()


def _bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
