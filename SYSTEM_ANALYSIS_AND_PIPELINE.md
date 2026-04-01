# RAG-1 System Analysis and Data Pipeline

## 1. Executive Summary

This project is a local hybrid retrieval system for e-commerce product search. It combines:

1. TF-IDF lexical retrieval
2. BM25 lexical-probabilistic retrieval
3. Dense semantic retrieval with sentence embeddings

The three ranked outputs are merged using Reciprocal Rank Fusion (RRF). Query parsing extracts structured constraints (category, color, budget, brand/model hints), and budget is applied as a post-filter.

Main orchestrator: `rag.py`

Supporting modules:

- `database.py`
- `tfidf_engine.py`
- `bm25_engine.py`
- `semantic_engine.py`
- `query_parser.py`
- `fusion.py`

---

## 2. Component-by-Component Analysis

## 2.1 Data Layer (`database.py`)

### Responsibilities

- Defines a fixed in-code catalog of 30 products.
- Creates SQLite database (`products.db`) with table `products`.
- Loads products into Python dictionaries.
- Builds canonical searchable text via `product_to_text`.

### Key behavior

- `create_db()` currently drops and recreates the table every time it is called.
- `load_products()` auto-creates DB if missing.
- `product_to_text()` repeats title twice to increase lexical weight.

### Strengths

- Simple, deterministic local dataset.
- Easy to inspect and debug.

### Risks

- Data reset on each startup if `create_db()` is always called.
- Not designed for incremental updates.

---

## 2.2 TF-IDF Engine (`tfidf_engine.py`)

### Responsibilities

- Fits a sparse TF-IDF matrix over product text.
- Retrieves top-k via cosine similarity.

### Configuration

- `ngram_range=(1,2)` enables unigrams and bigrams.
- `sublinear_tf=True` damps repeated-term frequency.
- `stop_words="english"` removes common English terms.

### Strengths

- Fast and effective for exact/near-exact keyword matching.
- Bigrams improve phrase retrieval (example: "noise cancelling").

### Limits

- Lexical dependence: synonyms without token overlap can be missed.

---

## 2.3 BM25 Engine (`bm25_engine.py`)

### Responsibilities

- Custom BM25 implementation from scratch.
- Uses manual tokenizer + custom stopword list.
- Scores each document with BM25 formula.

### Strengths

- Better term-frequency saturation and length normalization than plain TF-IDF.
- Robust for sparse keyword queries.

### Limits

- Tokenization is simple regex-based and language-limited.
- Query-document scoring loops over entire corpus (fine for 30 docs, less ideal at large scale).

---

## 2.4 Semantic Engine (`semantic_engine.py`)

### Responsibilities

- Loads SentenceTransformer model `all-MiniLM-L6-v2`.
- Encodes all products into 384-d vectors.
- Retrieves top-k by vector similarity.

### Design notes

- Product text includes synthetic price semantics labels:
  - budget/cheap/affordable for low price
  - mid range for medium price
  - premium/high end/flagship/professional for high price

### Strengths

- Captures semantic similarity beyond exact term overlap.
- Helps natural language queries that are less keyword-specific.

### Limits and caution

- Query path uses dot product and assumes L2 normalization behavior.
- Model loads at import time, which can increase startup latency.

---

## 2.5 Query Parser (`query_parser.py`)

### Responsibilities

- Converts natural language query to structured fields:
  - `category`
  - `model`
  - `color`
  - `budget`
  - metadata fields (`satisfied`, `reason`, `clarification_question`)

### Modes

1. Rule-based parser (default).
2. Anthropic API parser when `ANTHROPIC_API_KEY` is set.

### Strengths

- Adds structure before retrieval.
- Supports budget extraction and simple intent mapping.

### Risks

- Rule mapping can be brittle.
- Budget regex may capture unintended numbers.
- Brand list contains typo variants (`oneplug`, `oneplubs`) instead of `oneplus`.

---

## 2.6 Fusion and Diagnostics (`fusion.py`)

### Responsibilities

- Merges ranked lists with Reciprocal Rank Fusion (RRF).
- Normalizes scores for display diagnostics.
- Prints method-by-method similarity table.

### RRF formula

RRF(d) = sum over retrievers i of 1 / (k + rank_i(d))

with default smoothing constant `k=60`.

### Strengths

- Does not require calibrating raw score scales across engines.
- Stable and effective for hybrid search.

---

## 3. End-to-End Data Pipeline

## 3.1 Startup and Index Build

1. `rag.py` main block runs.
2. `create_db()` is called (recreates table and inserts 30 rows).
3. `build_rag()` loads products from SQLite.
4. `TFIDFEngine.fit()` builds sparse matrix.
5. `BM25Engine.fit()` builds token corpus, avgdl, idf map.
6. `SemanticEngine.fit()` builds dense embeddings for all products.
7. RAG runtime object is returned:
   - `tfidf`
   - `bm25`
   - `semantic`
   - `products`

## 3.2 Query-Time Pipeline

1. Receive user query.
2. Parse query into structured fields (`parse_query`).
3. Augment query text with parsed category/color/model.
4. Retrieve candidates from all three engines (top 10 each).
5. If budget exists, apply post-filter on each list (`price <= budget`).
6. Fuse ranked lists with RRF to produce final top-k.
7. Render diagnostics table with normalized per-engine scores.
8. Print final ranked product list.

---

## 4. Data Contracts

## 4.1 Product Record

A product dictionary has keys:

- `id`
- `title`
- `description`
- `manufacturer`
- `price`
- `category`
- `color`

## 4.2 Retrieval Result

Each engine returns list entries shaped as:

- `product`: product dict
- `score`: float
- `rank`: integer rank starting at 1

Fusion output returns:

- `product`
- `rrf_score`
- `rank`

## 4.3 Parsed Query Schema

- `satisfied`: bool
- `reason`: str or null
- `category`: str or null
- `model`: str or null
- `color`: str or null
- `budget`: float or null
- `clarification_question`: str or null

---

## 5. Runtime and Complexity Notes

Let:

- N = number of products
- D = embedding dimension (384)
- Q = number of query tokens

### Build phase

- TF-IDF fit: proportional to corpus size and vocabulary.
- BM25 fit: proportional to total token count.
- Semantic fit: transformer inference for N documents.

### Query phase

- TF-IDF: sparse cosine against N docs.
- BM25: scores each doc, approximately O(N * Q).
- Semantic: dense similarity approximately O(N * D).
- Fusion: linear in total candidate count from retrievers.

With N=30 this is very fast on local hardware.

---

## 6. What Is Good About This Design

1. Hybrid retrieval is the right strategy for practical recall.
2. RRF avoids fragile cross-model score normalization.
3. Query parser introduces useful business constraints.
4. Small, clear modular files make debugging straightforward.
5. Diagnostic table improves observability and explainability.

---

## 7. Gaps and Improvement Opportunities

1. Data lifecycle
- Startup recreates DB every run; move to idempotent initialization.

2. Semantic scoring correctness
- Explicitly normalize query and doc embeddings if cosine is intended.

3. Parser quality
- Fix brand typo for OnePlus.
- Tighten budget extraction to avoid accidental number capture.

4. Rank consistency after filter
- Recompute ranks after budget filtering before RRF.

5. Packaging consistency
- `pyproject.toml` currently has empty dependencies while `requirements.txt` contains real dependencies.

6. Scalability
- For larger corpora, add ANN indexing (for dense retrieval) and inverted index optimizations.

---

## 8. Recommended Future Pipeline (Production-Oriented)

1. Separate ingest and serve workflows.
2. Persist precomputed embeddings and lexical indexes.
3. Use strict schema validation for parser output.
4. Add query logging and relevance evaluation set.
5. Add metrics dashboard:
   - Recall@k
   - MRR
   - nDCG@k
   - latency percentiles

---

## 9. Practical Testing Checklist

1. Functional tests
- Query returns non-empty results for known intents.
- Budget filter excludes over-budget products.
- Category and color hints influence ranking.

2. Regression tests
- Snapshot top-k results for fixed test queries.
- Alert on major rank drift when changing parser/retrievers.

3. Robustness tests
- Empty query.
- Numeric-only query.
- Ambiguous query.
- Misspelled brand/category query.

---

## 10. Quick Pipeline Diagram (Text)

User Query
-> Query Parser (structured fields)
-> Query Augmentation (category/color/model)
-> Parallel Retrieval:
   - TF-IDF
   - BM25
   - Semantic
-> Budget Filter
-> Reciprocal Rank Fusion
-> Similarity Diagnostics Table
-> Final Top-K Results

---

## 11. Conclusion

The system is a strong educational and practical baseline for hybrid product search. The architecture is clean and the retrieval strategy is solid. The highest-impact improvements are parser hardening, semantic-score normalization clarity, rank recalculation after filtering, and separation of initialization from serving.
