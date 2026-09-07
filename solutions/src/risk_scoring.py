"""
risk_scoring.py
----------------
Feature-engineers per-customer transaction behavior and scores it for
suspicious activity using an Isolation Forest anomaly detector, with the
experiment tracked in MLflow (params, metrics, and the fitted model as
an artifact).

Engineered features specifically target known typologies:
  - structuring_score:     count of transactions just under the $10k reporting
                            threshold, clustered in a short window
  - velocity_score:        transaction count within short time windows
  - high_risk_country_pct: share of transaction volume tied to flagged
                            jurisdictions
  - avg_amount / txn_count / total_amount: baseline behavioral features
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HIGH_RISK_COUNTRIES = ["Country_X", "Country_Y", "Country_Z"]
STRUCTURING_THRESHOLD = 10_000
STRUCTURING_BAND = 1_500  # amounts within [threshold - band, threshold) count as "near-threshold"


def engineer_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df["is_near_threshold"] = transactions_df["amount"].between(
        STRUCTURING_THRESHOLD - STRUCTURING_BAND, STRUCTURING_THRESHOLD
    )
    transactions_df["is_high_risk_country"] = transactions_df["counterparty_country"].isin(
        HIGH_RISK_COUNTRIES
    )

    grouped = transactions_df.groupby("customer_id")

    features = grouped.agg(
        txn_count=("txn_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        near_threshold_count=("is_near_threshold", "sum"),
        high_risk_country_pct=("is_high_risk_country", "mean"),
    ).reset_index()

    # Velocity: max number of transactions by the same customer within any 3-hour window
    velocity = []
    for cid, group in grouped:
        times = group["timestamp"].sort_values().values.astype("datetime64[m]")
        max_count = 0
        for t in times:
            window_count = np.sum((times >= t) & (times <= t + np.timedelta64(180, "m")))
            max_count = max(max_count, window_count)
        velocity.append({"customer_id": cid, "max_txns_in_3h": max_count})
    velocity_df = pd.DataFrame(velocity)

    features = features.merge(velocity_df, on="customer_id", how="left")
    features["structuring_score"] = features["near_threshold_count"] / features["txn_count"]

    return features


def train_and_score(features_df: pd.DataFrame, contamination: float = 0.08):
    """Train an Isolation Forest on behavioral features and score every customer.
    Logs the run (params, metrics, model artifact) to MLflow."""

    feature_cols = [
        "txn_count", "total_amount", "avg_amount", "max_amount",
        "near_threshold_count", "high_risk_country_pct",
        "max_txns_in_3h", "structuring_score",
    ]
    X = features_df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlflow.set_experiment("aml-transaction-risk-scoring")
    with mlflow.start_run(run_name="isolation_forest_v1"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_cols", ",".join(feature_cols))

        model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=200
        )
        model.fit(X_scaled)

        raw_scores = model.decision_function(X_scaled)  # higher = more normal
        risk_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
        predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        features_df = features_df.copy()
        features_df["risk_score"] = risk_score.round(3)
        features_df["is_anomaly"] = predictions == -1

        flagged_pct = features_df["is_anomaly"].mean()
        mlflow.log_metric("pct_flagged", flagged_pct)
        mlflow.log_metric("avg_risk_score_flagged", features_df.loc[features_df["is_anomaly"], "risk_score"].mean())
        mlflow.log_metric("avg_risk_score_clean", features_df.loc[~features_df["is_anomaly"], "risk_score"].mean())

        mlflow.sklearn.log_model(model, "isolation_forest_model")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")

    return features_df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def main():
    transactions_df = pd.read_csv("data/transactions.csv")
    features_df = engineer_features(transactions_df)
    scored_df = train_and_score(features_df)
    scored_df.to_csv("data/transaction_risk_scores.csv", index=False)

    print(f"Customers scored: {len(scored_df)}")
    print(f"Flagged as anomalous: {scored_df['is_anomaly'].sum()}")
    print(scored_df[["customer_id", "risk_score", "is_anomaly", "structuring_score", "max_txns_in_3h"]].head(10))


if __name__ == "__main__":
    main()
