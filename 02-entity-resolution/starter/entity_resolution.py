"""
entity_resolution.py
---------------------
Fuzzy name matching against a watchlist, now combined with a second
identifying signal (date of birth) to distinguish real matches from
common-name coincidences.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz


def embed_names(names: list, vectorizer: TfidfVectorizer = None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        embeddings = vectorizer.fit_transform(names)
    else:
        embeddings = vectorizer.transform(names)
    return embeddings, vectorizer


def build_watchlist_index(watchlist_df: pd.DataFrame, n_neighbors: int = 1):
    embeddings, vectorizer = embed_names(watchlist_df["entity_name"].tolist())
    index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    index.fit(embeddings)
    return index, vectorizer


def resolve_entities(
    customers_df: pd.DataFrame,
    watchlist_df: pd.DataFrame,
    similarity_threshold: float = 0.70,
) -> pd.DataFrame:
    index, vectorizer = build_watchlist_index(watchlist_df)
    cust_embeddings, _ = embed_names(customers_df["customer_name"].tolist(), vectorizer)
    distances, indices = index.kneighbors(cust_embeddings)

    results = []
    for i, cust_row in enumerate(customers_df.itertuples()):
        nearest_idx = indices[i][0]
        cosine_sim = 1 - distances[i][0]
        wl_row = watchlist_df.iloc[nearest_idx]
        fuzzy_score = fuzz.token_sort_ratio(cust_row.customer_name, wl_row["entity_name"]) / 100.0
        blended_score = 0.6 * cosine_sim + 0.4 * fuzzy_score

        name_matches = blended_score >= similarity_threshold
        dob_matches = str(cust_row.date_of_birth) == str(wl_row["date_of_birth"])

        if name_matches and dob_matches:
            confidence = "high"        # same name AND same birth date -> likely the same person
        elif name_matches and not dob_matches:
            confidence = "review"      # name matches but DOB doesn't -> probably just a common name
        else:
            confidence = "low"         # name doesn't match well enough to matter

        results.append({
            "customer_id": cust_row.customer_id,
            "customer_name": cust_row.customer_name,
            "matched_watchlist_id": wl_row["watchlist_id"],
            "matched_watchlist_name": wl_row["entity_name"],
            "matched_risk_category": wl_row["risk_category"],
            "cosine_similarity": round(cosine_sim, 3),
            "fuzzy_score": round(fuzzy_score, 3),
            "blended_score": round(blended_score, 3),
            "dob_match": dob_matches,
            "confidence": confidence,
            "flagged": confidence in ("high", "review"),
        })
    return pd.DataFrame(results).sort_values("blended_score", ascending=False).reset_index(drop=True)


def main():
    customers_df = pd.read_csv("data/customers.csv")
    watchlist_df = pd.read_csv("data/watchlist.csv")

    matches_df = resolve_entities(customers_df, watchlist_df)
    matches_df.to_csv("data/entity_matches.csv", index=False)

    high_conf = matches_df[matches_df["confidence"] == "high"]
    review = matches_df[matches_df["confidence"] == "review"]
    print(f"High confidence (true matches): {len(high_conf)}")
    print(f"Needs review (likely common names): {len(review)}")

    common_name_ids = set(customers_df.loc[customers_df["is_common_name_collision_for_testing"], "customer_id"])
    high_conf_ids = set(high_conf["customer_id"])
    wrongly_escalated = common_name_ids & high_conf_ids
    print(f"Common-name customers wrongly marked 'high confidence': {len(wrongly_escalated)} (should be 0)")

    truth = set(customers_df.loc[customers_df["is_planted_alias_for_testing"], "customer_id"])
    caught = truth & high_conf_ids
    print(f"True planted aliases correctly caught as 'high confidence': {len(caught)} / {len(truth)}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()