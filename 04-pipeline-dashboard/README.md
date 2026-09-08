# Stage 4: Pipeline & Dashboard — Done

## The concept

### Orchestration

I had three separate scripts that each read/write CSVs. A pipeline is just the discipline of running them in the right order and combining their outputs into something a human actually wants to look at — in this case, a single case list where each customer has a priority score combining both signals (watchlist match + transaction anomaly).

I treated this as a small-scale version of what tools like Airflow or Databricks Workflows do at production scale: define steps, define dependencies between them, run in order, handle the handoffs.

### Streamlit — building a UI without knowing frontend

Streamlit turns a plain Python script into a web app by re-running the entire script top-to-bottom every time someone interacts with a widget. No JavaScript, no HTML — I just wrote things like `st.dataframe(df)` or `st.selectbox(...)` and it rendered. This turned out to be genuinely one of the fastest ways to build an internal tool, and analysts/investigators are exactly the audience this style of tool is built for.

## What I built

### `pipeline.py`

I opened `starter/pipeline.py`. This one was shorter — mostly about combining things I'd already built:

- Called my Stage 1-3 functions in order
- Merged the entity-resolution output and the risk-scoring output onto the customers table
- Computed a `case_priority` score (2 points for a watchlist flag, 1 point for a transaction anomaly flag, so a customer with both scores higher than either alone)
- Sorted and saved the combined case list

### `dashboard.py`

I opened `starter/dashboard.py`. This one had more scaffolding pre-built since it's UI code rather than a core concept — but I filled in:

- Loading and caching the data (`@st.cache_data`)
- Summary metric cards at the top (`st.metric`)
- A filterable case table
- A per-customer drill-down view showing their transaction history as a line chart

## Checkpoint

I ran this from the project root:

```bash
python3 04-pipeline-dashboard/starter/pipeline.py
streamlit run 04-pipeline-dashboard/starter/dashboard.py
```

I got a browser tab with a working dashboard showing my case queue. I clicked through a few customers and checked: the ones with the highest `case_priority` did actually correspond to my planted patterns from Stage 1. That confirmed I'd built a working (small-scale) financial-crime detection pipeline, end to end, and I understand every piece of it.

## What's next

I've now got a real, explainable project. A few directions to push it further once I'm comfortable:

- Swap the TF-IDF embeddings for a real sentence-transformer model and see how the precision/recall tradeoff changes
- Try a different anomaly detector (e.g. `LocalOutlierFactor`) and compare results in MLflow side-by-side
- Add a simple network-graph view: customers sharing an address or counterparty as a graph, which is a very real AML investigative technique
- Push it to GitHub with a clean README explaining my design choices — that write-up is often more valuable in an interview than the code itself
