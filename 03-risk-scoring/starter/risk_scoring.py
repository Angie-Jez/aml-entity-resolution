"""risk_scoring.py"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HIGH_RISK_COUNTRIES = ["Country_X", "Country_Y", "Country_Z"]
STRUCTURING_THRESHOLD = 10_000
STRUCTURING_BAND = 1_500


def engineer_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df["is_near_threshold"] = transactions_df["amount"].between(
        STRUCTURING_THRESHOLD - STRUCTURING_BAND, STRUCTURING_THRESHOLD)
    transactions_df["is_high_risk_country"] = transactions_df["counterparty_country"].isin(HIGH_RISK_COUNTRIES)

    grouped = transactions_df.groupby("customer_id")
    features = grouped.agg(
        txn_count=("txn_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        near_threshold_count=("is_near_threshold", "sum"),
        high_risk_country_pct=("is_high_risk_country", "mean"),
    ).reset_index()

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
    feature_cols = ["txn_count", "total_amount", "avg_amount", "max_amount",
                     "near_threshold_count", "high_risk_country_pct",
                     "max_txns_in_3h", "structuring_score"]
    X = features_df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    mlflow.set_experiment("aml-transaction-risk-scoring")
    with mlflow.start_run(run_name="isolation_forest_v1"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_features", len(feature_cols))

        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        model.fit(X_scaled)

        raw_scores = model.decision_function(X_scaled)
        risk_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
        predictions = model.predict(X_scaled)

        features_df = features_df.copy()
        features_df["risk_score"] = risk_score.round(3)
        features_df["is_anomaly"] = predictions == -1

        mlflow.log_metric("pct_flagged", features_df["is_anomaly"].mean())
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        print(f"MLflow run logged: {mlflow.active_run().info.run_id}")

    return features_df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def main():
    transactions_df = pd.read_csv("data/transactions.csv")
    features_df = engineer_features(transactions_df)
    scored_df = train_and_score(features_df)
    scored_df.to_csv("data/transaction_risk_scores.csv", index=False)
    print(f"Customers scored: {len(scored_df)}")
    print(f"Flagged as anomalous: {scored_df['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()