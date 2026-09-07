# AML Entity Resolution & Transaction Monitoring Pipeline

An end-to-end financial-crime detection pipeline I built from scratch —
combining vector-search entity resolution against a watchlist with
ML-based transaction anomaly scoring, tracked in MLflow and surfaced
through an analyst-facing Streamlit dashboard.

**What I built:**
- A synthetic data generator that plants realistic fraud/AML patterns
  (structuring, transaction velocity, watchlist aliases, and common-name
  collisions) so I could measure my own detection accuracy against a
  known ground truth
- A vector-search entity resolution engine — TF-IDF character embeddings
  and cosine-similarity nearest-neighbor search — that catches disguised
  name variants with 87% recall
- A second identity signal (date of birth) layered on top of name
  matching, specifically to solve a false-positive problem I'd
  encountered doing background-verification work: common names causing
  every coincidental match to get flagged as urgent
- An Isolation Forest anomaly-detection model trained on engineered
  behavioral features (structuring score, transaction velocity,
  high-risk-country exposure), with every experiment run tracked in MLflow
- A Streamlit dashboard so the output is something an analyst could
  actually use to review and prioritize cases, not just a spreadsheet
  of numbers

## Why I built it this way

Most ML tutorials use a dataset that's already clean and a problem
that's already labeled. Real fraud/AML work almost never looks like
that — you rarely get a labeled "this was fraud" column, and you have
to manufacture your own validation signal. So I planted known patterns
into my own synthetic data specifically so I could measure whether my
detection logic actually caught them, rather than just eyeballing the
results.

## Architecture
