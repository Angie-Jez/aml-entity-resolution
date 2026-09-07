"""
pipeline.py
-----------
End-to-end orchestration, now carrying the confidence-tier signal
(high / review / low) from entity resolution through to the final case list.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "01-synthetic-data", "starter"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "02-entity-resolution", "starter"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "03-risk-scoring", "starter"))

import pandas as pd

import generate_data
from entity_resolution import resolve_entities
from risk_scoring import engineer_features, train_and_score


def run_pipeline(regenerate_data: bool = True):
    if regenerate_data:
        generate_data.main()

    customers_df = pd.read_csv("data/customers.csv")
    watchlist_df = pd.read_csv("data/watchlist.csv")
    transactions_df = pd.read_csv("data/transactions.csv")

    print("\n--- Entity Resolution ---")
    entity_matches_df = resolve_entities(customers_df, watchlist_df)
    entity_matches_df.to_csv("data/entity_matches.csv", index=False)
    print(f"High confidence: {(entity_matches_df['confidence'] == 'high').sum()}")
    print(f"Needs review: {(entity_matches_df['confidence'] == 'review').sum()}")

    print("\n--- Transaction Risk Scoring ---")
    features_df = engineer_features(transactions_df)
    scored_df = train_and_score(features_df)
    scored_df.to_csv("data/transaction_risk_scores.csv", index=False)
    print(f"Flagged anomalous customers: {scored_df['is_anomaly'].sum()}")

    print("\n--- Combined Case List ---")
    case_list = customers_df.merge(
        entity_matches_df[["customer_id", "confidence", "dob_match",
                            "matched_watchlist_name", "blended_score"]]
        .rename(columns={"blended_score": "watchlist_score"}),
        on="customer_id", how="left",
    ).merge(
        scored_df[["customer_id", "is_anomaly", "risk_score", "structuring_score",
                   "max_txns_in_3h", "high_risk_country_pct"]]
        .rename(columns={"is_anomaly": "transaction_flag"}),
        on="customer_id", how="left",
    )

    case_list["confidence"] = case_list["confidence"].fillna("low")

    # Case priority: a high-confidence watchlist match is the most urgent signal,
    # "review" (likely a common name) is moderate, a transaction anomaly adds more weight
    confidence_points = case_list["confidence"].map({"high": 3, "review": 1, "low": 0})
    transaction_points = case_list["transaction_flag"].fillna(False).astype(int)
    case_list["case_priority"] = confidence_points + transaction_points

    case_list = case_list.sort_values("case_priority", ascending=False).reset_index(drop=True)
    case_list.to_csv("data/combined_case_list.csv", index=False)

    high_priority = case_list[case_list["case_priority"] >= 3]
    print(f"High-priority cases (confirmed watchlist match, any transaction signal): {len(high_priority)}")
    print("\nPipeline complete. Outputs written to data/.")

    return case_list


if __name__ == "__main__":
    run_pipeline()