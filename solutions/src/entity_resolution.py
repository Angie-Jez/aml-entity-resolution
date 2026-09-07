"""
entity_resolution.py
---------------------
Fuzzy name matching between the customer book and a sanctions/PEP watchlist,
implemented as a vector-search problem:

  1. Each name is embedded as a character n-gram TF-IDF vector (captures
     misspellings, transliteration drift, and partial-name matches that
     exact-string matching would miss).
  2. A nearest-neighbor index (cosine similarity) retrieves the closest
     watchlist entity for every customer.
  3. Matches above a similarity threshold are flagged for analyst review,
     with a similarity score to support triage/prioritization.

Swap-in note: in production this embedding step would typically use a
pretrained sentence-transformer or an LLM embedding endpoint, and the
nearest-neighbor index would run on a proper vector database (e.g., FAISS,
pgvector, or Databricks Vector Search) rather than in-memory sklearn.
The interface below is written so that swap is a one-function change
(see `embed_names`).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz


def embed_names(names: list[str], vectorizer: TfidfVectorizer = None):
    """
    Embed a list of names as character n-gram TF-IDF vectors.
    Returns (embeddings, fitted_vectorizer).

    Swap point: replace this function with a call to a sentence-transformer
    or embedding API to upgrade to semantic (rather than lexical) vector search.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        embeddings = vectorizer.fit_transform(names)
    else:
        embeddings = vectorizer.transform(names)
    return embeddings, vectorizer


def build_watchlist_index(watchlist_df: pd.DataFrame, n_neighbors: int = 1):
    """Fit the vector index over watchlist entity names."""
    embeddings, vectorizer = embed_names(watchlist_df["entity_name"].tolist())
    index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    index.fit(embeddings)
    return index, vectorizer


def resolve_entities(
    customers_df: pd.DataFrame,
    watchlist_df: pd.DataFrame,
    similarity_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    For every customer, retrieve the nearest watchlist entity via vector
    search and flag matches above `similarity_threshold`. Also computes a
    rapidfuzz token-sort ratio as a secondary corroborating signal, since
    a hybrid lexical + fuzzy score reduces false positives versus either
    method alone.
    """
    index, vectorizer = build_watchlist_index(watchlist_df)

    cust_embeddings, _ = embed_names(customers_df["customer_name"].tolist(), vectorizer)
    distances, indices = index.kneighbors(cust_embeddings)

    results = []
    for i, cust_row in enumerate(customers_df.itertuples()):
        nearest_idx = indices[i][0]
        cosine_sim = 1 - distances[i][0]
        wl_row = watchlist_df.iloc[nearest_idx]

        fuzzy_score = fuzz.token_sort_ratio(
            cust_row.customer_name, wl_row["entity_name"]
        ) / 100.0

        # Blended score: vector similarity carries more weight, fuzzy score corroborates
        blended_score = 0.6 * cosine_sim + 0.4 * fuzzy_score

        results.append({
            "customer_id": cust_row.customer_id,
            "customer_name": cust_row.customer_name,
            "matched_watchlist_id": wl_row["watchlist_id"],
            "matched_watchlist_name": wl_row["entity_name"],
            "matched_risk_category": wl_row["risk_category"],
            "cosine_similarity": round(cosine_sim, 3),
            "fuzzy_score": round(fuzzy_score, 3),
            "blended_score": round(blended_score, 3),
            "flagged": blended_score >= similarity_threshold,
        })

    return pd.DataFrame(results).sort_values("blended_score", ascending=False).reset_index(drop=True)


def main():
    customers_df = pd.read_csv("data/customers.csv")
    watchlist_df = pd.read_csv("data/watchlist.csv")

    matches_df = resolve_entities(customers_df, watchlist_df)
    matches_df.to_csv("data/entity_matches.csv", index=False)

    flagged = matches_df[matches_df["flagged"]]
    print(f"Total customers scored: {len(matches_df)}")
    print(f"Flagged for analyst review: {len(flagged)}")

    # Quick precision/recall check against the planted ground truth
    truth = set(
        customers_df.loc[customers_df["is_planted_alias_for_testing"], "customer_id"]
    )
    flagged_ids = set(flagged["customer_id"])
    tp = len(truth & flagged_ids)
    fp = len(flagged_ids - truth)
    fn = len(truth - flagged_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} (vs. planted ground truth)")


if __name__ == "__main__":
    main()
