"""
pipeline.py
-----------
End-to-end orchestration: generate data -> resolve entities against the
watchlist -> engineer transaction features -> score risk -> combine into
a single analyst-facing case list.

Run with: python3 -m src.pipeline
"""

import pandas as pd

from src import generate_data
from src.entity_resolution import resolve_entities
from src.risk_scoring import engineer_features, train_and_score


def run_pipeline(regenerate_data: bool = True):
    if regenerate_data:
        generate_data.main()

    customers_df = pd.read_csv("data/customers.csv")
    watchlist_df = pd.read_csv("data/watchlist.csv")
    transactions_df = pd.read_csv("data/transactions.csv")

    print("\n--- Entity Resolution ---")
    entity_matches_df = resolve_entities(customers_df, watchlist_df)
    entity_matches_df.to_csv("data/entity_matches.csv", index=False)
    print(f"Flagged watchlist matches: {entity_matches_df['flagged'].sum()}")

    print("\n--- Transaction Risk Scoring ---")
    features_df = engineer_features(transactions_df)
    scored_df = train_and_score(features_df)
    scored_df.to_csv("data/transaction_risk_scores.csv", index=False)
    print(f"Flagged anomalous customers: {scored_df['is_anomaly'].sum()}")

    print("\n--- Combined Case List ---")
    case_list = customers_df.merge(
        entity_matches_df[["customer_id", "flagged", "matched_watchlist_name", "blended_score"]]
        .rename(columns={"flagged": "watchlist_flag", "blended_score": "watchlist_score"}),
        on="customer_id", how="left",
    ).merge(
        scored_df[["customer_id", "is_anomaly", "risk_score", "structuring_score", "max_txns_in_3h", "high_risk_country_pct"]]
        .rename(columns={"is_anomaly": "transaction_flag"}),
        on="customer_id", how="left",
    )

    case_list["case_priority"] = (
        case_list["watchlist_flag"].fillna(False).astype(int) * 2
        + case_list["transaction_flag"].fillna(False).astype(int)
    )
    case_list = case_list.sort_values("case_priority", ascending=False).reset_index(drop=True)
    case_list.to_csv("data/combined_case_list.csv", index=False)

    high_priority = case_list[case_list["case_priority"] >= 2]
    print(f"High-priority cases (watchlist match AND/OR transaction anomaly): {len(high_priority)}")
    print("\nPipeline complete. Outputs written to data/.")

    return case_list


if __name__ == "__main__":
    run_pipeline()
