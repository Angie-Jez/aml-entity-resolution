# Stage 2: Entity Resolution (Vector Search, from scratch) — Done

"Vector search" sounds intimidating but the idea turned out to be simple: turn text into numbers (a vector) such that similar text produces similar numbers, then find nearest neighbors using regular math.

## Step 1: Turning names into vectors (TF-IDF over character n-grams)

I expected "vectorizing text" to mean one number per word, but I used something slightly different and more robust to misspellings: character n-grams. For the name "Jane", the character 3-grams (with padding) look like: " Ja", "Jan", "ane", "ne ". Every possible 3-4 character chunk becomes a dimension, and a name gets a 1 (or a weighted score, via TF-IDF) in the dimensions it contains, 0 elsewhere.

This beats word-level or exact matching because "Jane Smith" and "Jane Smyth" share almost all the same character chunks ("Jan", "ane", " Sm", "mit" vs "myt"...) even though they're different strings and different words. A word-level match would have completely missed this.

I let scikit-learn handle it:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
vectors = vectorizer.fit_transform(["Jane Smith", "Jane Smyth", "Bob Jones"])
```

vectors ended up as a matrix — one row per name, one column per n-gram seen across all names. Each row is that name's "vector."

## Step 2: Measuring similarity (cosine similarity)

Once two names are vectors, "how similar are they" becomes "how close are these two vectors, geometrically?" I used cosine similarity, which measures the angle between two vectors — 1.0 means identical direction (very similar), 0 means completely unrelated. I preferred it over raw distance here because it ignores vector length (which for TF-IDF often just reflects name length) and focuses on direction (which reflects content overlap).

## Step 3: Finding the closest match (nearest-neighbor search)

With a watchlist of only 25 names, I technically could have computed cosine similarity against all 25 for every customer by hand, in a loop — that's literally what "vector search" is at small scale. Instead I used sklearn's NearestNeighbors, which does exactly this, just efficiently. It's the same retrieval pattern production vector databases (FAISS, pgvector, Databricks Vector Search) use at massive scale, just with fancier indexing.

## Blending in a second, non-vector signal

I knew vector similarity alone can be fooled or too noisy, so I blended the cosine similarity score with a rapidfuzz fuzzy-string score (token_sort_ratio) as a second, independent signal — if two methods that work differently both agree a name is close, that's stronger evidence than either alone.

## What I built

I opened `starter/entity_resolution.py` and implemented:

- `embed_names(names, vectorizer)` — fits (or reuses) a TfidfVectorizer and returns the embeddings.
- `build_watchlist_index(watchlist_df)` — fits a NearestNeighbors index over the watchlist's embedded names.
- `resolve_entities(customers_df, watchlist_df, similarity_threshold)` — for every customer, finds their nearest watchlist neighbor, computes a blended score (60% cosine similarity + 40% fuzzy score), and flags matches above the threshold.

## Checkpoint — and a real ML skill: threshold tuning

I ran this from the project root:

```bash
python3 02-entity-resolution/starter/entity_resolution.py
```

It printed precision and recall against my planted ground truth from Stage 1. I tried a few different similarity_threshold values (0.5, 0.6, 0.7, 0.8) and watched precision and recall trade off against each other.

I treated this tradeoff as the actual job, not a side detail. A lower threshold catches more real matches (higher recall) but also flags more false alarms (lower precision) that an analyst has to manually clear. Since in AML/fraud work you almost always deliberately favor recall — a missed sanctions match is catastrophic, while a false positive just costs an analyst two minutes — I picked a threshold with that reasoning in mind, ready to explain why rather than just because it "looked good."
