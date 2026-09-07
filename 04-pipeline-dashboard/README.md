# Stage 4: Pipeline & Dashboard

## The concept

### Orchestration

Right now you have three separate scripts that each read/write CSVs. A
**pipeline** is just the discipline of running them in the right order and
combining their outputs into something a human actually wants to look at —
in this case, a single case list where each customer has a priority score
combining both signals (watchlist match + transaction anomaly).

This is a small-scale version of what tools like Airflow or Databricks
Workflows do at production scale: define steps, define dependencies between
them, run in order, handle the handoffs.

### Streamlit — building a UI without knowing frontend

Streamlit turns a plain Python script into a web app by re-running your
entire script top-to-bottom every time someone interacts with a widget. No
JavaScript, no HTML — you write things like `st.dataframe(df)` or
`st.selectbox(...)` and it renders. This is genuinely one of the fastest
ways to build an internal tool, and analysts/investigators are exactly the
audience this style of tool is built for.

## What you'll build

### `pipeline.py`

Open `starter/pipeline.py`. This one's shorter — mostly about combining
things you already built:

- Call your Stage 1-3 functions in order
- Merge the entity-resolution output and the risk-scoring output onto the
  customers table
- Compute a `case_priority` score (try: 2 points for a watchlist flag, 1
  point for a transaction anomaly flag, so a customer with both scores
  higher than either alone)
- Sort and save the combined case list

### `dashboard.py`

Open `starter/dashboard.py`. This one has more scaffolding pre-built since
it's UI code rather than a core concept — but you'll fill in:

- Loading and caching the data (`@st.cache_data`)
- Summary metric cards at the top (`st.metric`)
- A filterable case table
- A per-customer drill-down view showing their transaction history as a
  line chart

## Checkpoint

Run from the **project root**:

```bash
python3 04-pipeline-dashboard/starter/pipeline.py
streamlit run 04-pipeline-dashboard/starter/dashboard.py
```

You should get a browser tab with a working dashboard showing your case
queue. Click through a few customers and check: do the ones with the
highest `case_priority` actually correspond to your planted patterns from
Stage 1? If yes — you've built a working (small-scale) financial-crime
detection pipeline, end to end, and you understand every piece of it.

## Stuck?

Compare against `../../solutions/src/pipeline.py` and
`../../solutions/app/dashboard.py`.

## What's next

You've now got a real, explainable project. A few directions to push it
further once you're comfortable:

- Swap the TF-IDF embeddings for a real sentence-transformer model and
  see how the precision/recall tradeoff changes
- Try a different anomaly detector (e.g. `LocalOutlierFactor`) and compare
  results in MLflow side-by-side
- Add a simple network-graph view: customers sharing an address or
  counterparty as a graph, which is a very real AML investigative technique
- Push it to GitHub with a clean README explaining your design choices —
  that write-up is often more valuable in an interview than the code itself
