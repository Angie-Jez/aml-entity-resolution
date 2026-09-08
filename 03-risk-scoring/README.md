# Stage 3: Transaction Risk Scoring (Anomaly Detection + MLflow) — Done

## The concept

### Part A: Feature engineering

Raw transactions (`amount`, `timestamp`, `country`) aren't directly useful to a model — I needed to summarize each customer's behavior into a fixed set of numbers a model can compare across customers. This is called **feature engineering**, and it's where domain knowledge matters more than algorithm choice. For AML, I made sure the features I engineered directly targeted known typologies:

- **Structuring score**: what fraction of a customer's transactions sit suspiciously just under the $10,000 reporting threshold.
- **Velocity**: the most transactions this customer made within any 3-hour window. (Normal customers rarely transact rapidly; layering schemes often do.)
- **High-risk country %**: what fraction of transaction volume touches a flagged jurisdiction.
- **Baseline behavior**: total volume, average transaction size, max transaction.

A model is only as good as these features — a fancy algorithm on bad features loses to a simple algorithm on good features almost every time.

### Part B: Anomaly detection without labels (Isolation Forest)

I hit a problem you run into constantly in this domain: there's no column that says "this customer is committing fraud." Nobody hands you labeled fraud data (and if they did, you'd have already caught it). This called for **unsupervised learning** — finding structure without labels.

I used **Isolation Forest** as the default for this. The intuition: it builds many random decision trees that split the data on random features at random thresholds. Points that are normal (similar to most other points) take many splits to isolate into their own leaf. Points that are unusual (extreme in some feature, or unusually combined) get isolated in very few splits — they "stick out" and get separated quickly almost no matter how you randomly split. The algorithm scores every point by how quickly it gets isolated, on average, across many random trees. Short average path = high anomaly score.

I didn't implement this myself — `sklearn.ensemble.IsolationForest` handled it. My job was to feed it good features and correctly interpret the output.

### Part C: MLflow — treating ML experiments like an engineer, not a scientist doodling

I recognized that without a tool like MLflow, "experimenting" with a model usually means re-running a script, eyeballing the output, and — if you're not careful — forgetting which parameters produced which result. So I had MLflow log three things for every run, permanently, in a queryable place:

- **Params**: what I configured (contamination rate, features used...)
- **Metrics**: how it performed (% flagged, avg risk score...)
- **Artifacts**: the actual trained model file, so I can reload it later without retraining

This matters more as projects grow — six months from now I won't remember which contamination value I used, but MLflow will.

## What I built

I opened `starter/risk_scoring.py` and implemented:

- `engineer_features(transactions_df)` — groups transactions by customer and computes the features described above (the velocity feature was the trickiest, per the hint in the TODO).
- `train_and_score(features_df, contamination)` — scales features, fits an `IsolationForest`, scores every customer, and logs the run to MLflow.

## Checkpoint

I ran this from the project root:

```bash
python3 03-risk-scoring/starter/risk_scoring.py
```

Then browsed my logged experiment:

```bash
mlflow ui
```

I opened the URL it printed (usually `http://localhost:5000`) and looked at my run — I could see the params, metrics, and the model artifact I'd logged. I tried changing `contamination` (e.g. 0.05 vs 0.15) and re-running — a second run appeared in the MLflow UI that I could compare against the first.

## Stuck?

I compared against `../../solutions/src/risk_scoring.py` when needed.

## Next

→ [Stage 4: Pipeline & Dashboard](../04-pipeline-dashboard/README.md)
