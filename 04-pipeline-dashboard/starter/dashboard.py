"""
dashboard.py
------------
Analyst dashboard, now showing watchlist match confidence (high/review/low)
instead of a flat true/false flag.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AML Case Review Dashboard", layout="wide")
DATA_DIR = "data"


@st.cache_data
def load_data():
    case_list = pd.read_csv(os.path.join(DATA_DIR, "combined_case_list.csv"))
    entity_matches = pd.read_csv(os.path.join(DATA_DIR, "entity_matches.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    return case_list, entity_matches, transactions


st.title("AML Entity Resolution & Transaction Monitoring")
st.caption("Vector-search entity resolution, disambiguated with date of birth, "
           "combined with MLflow-tracked anomaly scoring.")

try:
    case_list, entity_matches, transactions = load_data()
except FileNotFoundError:
    st.error("No pipeline output found. Run pipeline.py first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(case_list))
col2.metric("High Confidence Matches", int((case_list["confidence"] == "high").sum()))
col3.metric("Needs Review (common names)", int((case_list["confidence"] == "review").sum()))
col4.metric("High-Priority Cases", int((case_list["case_priority"] >= 3).sum()))

st.divider()
st.subheader("Case Queue")

priority_filter = st.select_slider("Minimum case priority", options=[0, 1, 2, 3, 4], value=1)
filtered = case_list[case_list["case_priority"] >= priority_filter]

display_cols = ["customer_id", "customer_name", "country", "case_priority",
                 "confidence", "matched_watchlist_name", "watchlist_score", "dob_match",
                 "transaction_flag", "risk_score", "structuring_score"]
st.dataframe(filtered[display_cols], use_container_width=True, height=400)

st.divider()
st.subheader("Investigate a Customer")
selected_id = st.selectbox("Select customer ID", options=filtered["customer_id"].tolist())
if selected_id:
    cust_txns = transactions[transactions["customer_id"] == selected_id].sort_values("timestamp")
    st.dataframe(cust_txns, use_container_width=True, height=250)
    if len(cust_txns):
        st.line_chart(cust_txns.set_index("timestamp")["amount"])