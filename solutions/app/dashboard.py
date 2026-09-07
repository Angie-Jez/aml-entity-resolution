"""
dashboard.py
------------
Analyst-facing Streamlit dashboard for reviewing flagged AML cases.
Run with: streamlit run app/dashboard.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="AML Case Review Dashboard", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_data():
    case_list = pd.read_csv(os.path.join(DATA_DIR, "combined_case_list.csv"))
    entity_matches = pd.read_csv(os.path.join(DATA_DIR, "entity_matches.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    return case_list, entity_matches, transactions


st.title("🔎 AML Entity Resolution & Transaction Monitoring")
st.caption(
    "Synthetic-data demo pipeline: vector-search entity resolution against a "
    "watchlist, combined with ML-based transaction anomaly scoring (MLflow-tracked)."
)

try:
    case_list, entity_matches, transactions = load_data()
except FileNotFoundError:
    st.error("No pipeline output found. Run `python3 -m src.pipeline` first to generate data.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(case_list))
col2.metric("Watchlist Matches", int(case_list["watchlist_flag"].fillna(False).sum()))
col3.metric("Transaction Anomalies", int(case_list["transaction_flag"].fillna(False).sum()))
col4.metric("High-Priority Cases", int((case_list["case_priority"] >= 2).sum()))

st.divider()

st.subheader("Case Queue")
priority_filter = st.select_slider(
    "Minimum case priority", options=[0, 1, 2, 3], value=1,
    help="0 = all customers, 3 = watchlist match AND transaction anomaly"
)
filtered = case_list[case_list["case_priority"] >= priority_filter].copy()

display_cols = [
    "customer_id", "customer_name", "country", "case_priority",
    "watchlist_flag", "matched_watchlist_name", "watchlist_score",
    "transaction_flag", "risk_score", "structuring_score", "max_txns_in_3h",
]
st.dataframe(
    filtered[display_cols].style.background_gradient(
        subset=["case_priority"], cmap="Reds"
    ),
    use_container_width=True,
    height=400,
)

st.divider()

st.subheader("Investigate a Customer")
selected_id = st.selectbox("Select customer ID", options=filtered["customer_id"].tolist())

if selected_id:
    cust = case_list[case_list["customer_id"] == selected_id].iloc[0]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Customer Profile**")
        st.json({
            "customer_id": cust["customer_id"],
            "name": cust["customer_name"],
            "country": cust["country"],
            "watchlist_flag": bool(cust["watchlist_flag"]) if pd.notna(cust["watchlist_flag"]) else False,
            "matched_watchlist_entity": cust.get("matched_watchlist_name", None),
            "watchlist_similarity": cust.get("watchlist_score", None),
        })

    with c2:
        st.markdown("**Transaction History**")
        cust_txns = transactions[transactions["customer_id"] == selected_id].sort_values("timestamp")
        st.dataframe(cust_txns, use_container_width=True, height=250)
        if len(cust_txns):
            st.line_chart(cust_txns.set_index("timestamp")["amount"])

st.divider()
st.caption(
    "Built as a portfolio project demonstrating entity resolution (vector search), "
    "MLflow-tracked anomaly detection, and analyst-facing tooling for financial-crime "
    "investigations. All data is synthetic."
)
