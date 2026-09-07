# Stage 3: Transaction Risk Scoring (Anomaly Detection + MLflow)

## The concept

### Part A: Feature engineering

Raw transactions (`amount`, `timestamp`, `country`) aren't directly useful
to a model — you need to summarize each *customer's* behavior into a fixed
set of numbers a model can compare across customers. This is called
**feature engineering**, and it's where domain knowledge matters more
than algorithm choice. For AML, the features you engineer should directly
target known typologies:

- **Structuring score**: what fraction of a customer's transactions sit
  suspiciously just under the $10,000 reporting threshold?
- **Velocity**: what's the most transactions this customer made within
  any 3-hour window? (Normal customers rarely transact rapidly; layering
  schemes often do.)
- **High-risk country %**: what fraction of transaction volume touches a
  flagged jurisdiction?
- Baseline behavior: total volume, average transaction size, max transaction

A model is only as good as these features — a fancy algorithm on bad
features loses to a simple algorithm on good features almost every time.

### Part B: Anomaly detection without labels (Isolation Forest)

Here's a problem you'll hit constantly in this domain: **you don't have a
column that says "this customer is committing fraud."** Nobody hands you
labeled fraud data (and if they did, you'd have already caught it). This
is called **unsupervised learning** — finding structure without labels.

**Isolation Forest** is a good default for this. The intuition: it builds
many random decision trees that split the data on random features at
random thresholds. Points that are *normal* (similar to most other points)
take many splits to isolate into their own leaf. Points that are *unusual*
(extreme in some feature, or unusually combined) get isolated in very few
splits — they "stick out" and get separated quickly almost no matter how
you randomly split. The algorithm scores every point by how quickly it
gets isolated, on average, across many random trees. Short average path =
high anomaly score.

You don't need to implement this yourself — `sklearn.ensemble.IsolationForest`
does it. Your job is to feed it good features and correctly interpret the
output.

### Part C: MLflow — treating ML experiments like an engineer, not a scientist doodling

Without a tool like MLflow, "experimenting" with a model usually means
re-running a script, eyeballing the output, and — if you're not careful —
forgetting which parameters produced which result. MLflow logs three things
for every run, permanently, in a queryable place:

- **Params**: what did you configure? (contamination rate, features used...)
- **Metrics**: how did it perform? (% flagged, avg risk score...)
- **Artifacts**: the actual trained model file, so you can reload it later
  without retraining

This matters more as projects grow — six months from now you won't
remember which contamination value you used, but MLflow will.

## What you'll build

Open `starter/risk_scoring.py`:

- `engineer_features(transactions_df)` — group transactions by customer,
  compute the features described above (the velocity feature is the
  trickiest — see the hint in the TODO)
- `train_and_score(features_df, contamination)` — scale features, fit an
  `IsolationForest`, score every customer, and log the run to MLflow

## Checkpoint

Run from the **project root**:

```bash
python3 03-risk-scoring/starter/risk_scoring.py
```

Then browse your logged experiment:

```bash
mlflow ui
```

Open the URL it prints (usually `http://localhost:5000`) and look at your
run — you should see the params, metrics, and the model artifact you logged.
Try changing `contamination` (e.g. 0.05 vs 0.15) and re-running — you should
see a second run appear in the MLflow UI you can compare against the first.

## Stuck?

Compare against `../../solutions/src/risk_scoring.py`.

## Next

→ [Stage 4: Pipeline & Dashboard](../04-pipeline-dashboard/README.md)
