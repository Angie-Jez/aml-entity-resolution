# Stage 2: Entity Resolution (Vector Search, from scratch)

## The concept

"Vector search" sounds intimidating but the idea is simple: **turn text
into numbers (a vector) such that similar text produces similar numbers,
then find nearest neighbors using regular math.**

### Step 1: Turning names into vectors (TF-IDF over character n-grams)

You might expect "vectorizing text" to mean one number per word. We're
going to do something slightly different and more robust to misspellings:
**character n-grams**. For the name "Jane", the character 3-grams (with
padding) look like: `" Ja"`, `"Jan"`, `"ane"`, `"ne "`. Every possible
3-4 character chunk becomes a dimension, and a name gets a 1 (or a
weighted score, via TF-IDF) in the dimensions it contains, 0 elsewhere.

**Why this beats word-level or exact matching:** "Jane Smith" and "Jane
Smyth" share almost all the same character chunks (`"Jan"`, `"ane"`,
`" Sm"`, `"mit"` vs `"myt"`...) even though they're different strings and
different words. A word-level match would completely miss this.

`scikit-learn` does this for you:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
vectors = vectorizer.fit_transform(["Jane Smith", "Jane Smyth", "Bob Jones"])
```

`vectors` is now a matrix — one row per name, one column per n-gram seen
across all names. Each row is that name's "vector."

### Step 2: Measuring similarity (cosine similarity)

Once two names are vectors, "how similar are they" becomes "how close
are these two vectors, geometrically?" **Cosine similarity** measures the
angle between two vectors — 1.0 means identical direction (very similar),
0 means completely unrelated. It's preferred over raw distance here
because it ignores vector *length* (which for TF-IDF often just reflects
name length) and focuses on *direction* (which reflects content overlap).

### Step 3: Finding the closest match (nearest-neighbor search)

With a watchlist of 25 names, you technically could compute cosine
similarity against all 25 for every customer, by hand, in a loop. That's
literally what "vector search" is at small scale. `sklearn`'s
`NearestNeighbors` does exactly this, just efficiently — this is the same
retrieval pattern production vector databases (FAISS, pgvector, Databricks
Vector Search) use at massive scale, just with fancier indexing.

## Why blend in a second, non-vector signal

Vector similarity alone can be fooled or too noisy. This project blends
the cosine similarity score with a `rapidfuzz` fuzzy-string score
(`token_sort_ratio`) as a second, independent signal — if two methods
that work differently both agree a name is close, that's stronger evidence
than either alone.

## What you'll build

Open `starter/entity_resolution.py`:

- `embed_names(names, vectorizer)` — fit (or reuse) a `TfidfVectorizer`
  and return the embeddings
- `build_watchlist_index(watchlist_df)` — fit a `NearestNeighbors` index
  over the watchlist's embedded names
- `resolve_entities(customers_df, watchlist_df, similarity_threshold)` —
  for every customer, find their nearest watchlist neighbor, compute a
  blended score (60% cosine similarity + 40% fuzzy score), and flag matches
  above the threshold

## Checkpoint — and a real ML skill: threshold tuning

Run from the **project root**:

```bash
python3 02-entity-resolution/starter/entity_resolution.py
```

This should print precision and recall against your planted ground truth
from Stage 1. Try a few different `similarity_threshold` values (0.5, 0.6,
0.7, 0.8) and watch precision and recall trade off against each other.

**This tradeoff is the actual job, not a side detail.** A lower threshold
catches more real matches (higher recall) but also flags more false alarms
(lower precision) that an analyst has to manually clear. In AML/fraud work
you almost always deliberately favor recall — a missed sanctions match is
catastrophic; a false positive costs an analyst two minutes. Pick a
threshold and be ready to explain *why* you picked it, not just that it
"looked good."

## Stuck?

Compare against `../../solutions/src/entity_resolution.py`.

## Next

→ [Stage 3: Risk Scoring](../03-risk-scoring/README.md)
