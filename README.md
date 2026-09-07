# Learn: AML Entity Resolution & Transaction Monitoring Pipeline

A hands-on tutorial for building an AML/financial-crime detection pipeline
from scratch — the same project structure used in a real portfolio piece,
but broken into stages you build yourself.

**You'll come out knowing:**
- How "vector search" actually works under the hood (no magic — it's just
  math you can write yourself)
- How to engineer features that capture real-world fraud/AML behavior patterns
- How anomaly detection works when you don't have labeled "fraud" examples
- How to track ML experiments properly with MLflow instead of guessing
- How to wire it all into a dashboard a non-technical analyst could use

**Assumed background:** solid Python (functions, classes, pandas basics).
No prior ML or data science experience assumed — every concept is explained
before you're asked to use it.

## How this works

Each stage has:
1. A `README.md` explaining the concept in plain language, with a diagram
   or worked example where it helps
2. A `starter/` file with function signatures and `# TODO` comments —
   you write the logic
3. A "checkpoint" — a command to run that tells you if your code is working
4. A pointer to `solutions/` if you get stuck (try for at least 15-20
   minutes before peeking — that's where the learning happens)

## Step 0: Tools & environment (do this before anything else)

**Editor:** [VS Code](https://code.visualstudio.com/) (free) is the easiest
choice — install it, then install its official "Python" extension
(search in the Extensions sidebar, the icon with four squares). Any editor
works, but VS Code's built-in terminal and Python integration make the
steps below simpler.

**Terminal:** you'll run everything from a command-line terminal, not by
double-clicking files.
- **Mac:** open the Terminal app (Cmd+Space, type "Terminal")
- **Windows:** open VS Code, then use its built-in terminal (menu:
  Terminal → New Terminal) — this avoids Windows-specific path issues
- **Linux:** whatever terminal you normally use

**Check Python is installed** (need 3.9+):
```bash
python3 --version
```
If that fails, install from [python.org](https://www.python.org/downloads/)
(Windows users: check "Add Python to PATH" during install).

**Open the project:**
1. Unzip this folder wherever you keep code projects
2. In VS Code: File → Open Folder → select the unzipped `learn-aml-pipeline` folder
3. Open the terminal inside VS Code (Terminal → New Terminal) — it should
   already be sitting in the project folder

**Now run setup** (this is the same as below, just repeated here so it's
in the right order):
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
You'll know it worked if your terminal prompt now shows `(venv)` at the
start of the line, and `pip install` finishes with no red error text.

**Every time you come back to work on this** (new terminal session), you
only need to re-run the `source venv/bin/activate` line — not the whole
setup — then you're ready to pick up where you left off.

Once that's done, go to [Stage 1](01-synthetic-data/README.md) and start writing code.

## The path

| Stage | What you'll build | Core concept |
|---|---|---|
| [01 – Synthetic Data](01-synthetic-data/README.md) | A fake bank's customers, watchlist, and transactions | Why synthetic data needs *planted patterns*, not just randomness |
| [02 – Entity Resolution](02-entity-resolution/README.md) | Fuzzy name matching against a watchlist | TF-IDF vectors, cosine similarity, nearest-neighbor search |
| [03 – Risk Scoring](03-risk-scoring/README.md) | A model that flags suspicious transaction behavior | Feature engineering, unsupervised anomaly detection, MLflow |
| [04 – Pipeline & Dashboard](04-pipeline-dashboard/README.md) | Wiring it together + an analyst UI | Orchestration, Streamlit basics |

Work through them in order — each stage's output feeds the next.

## Why this project (not a toy dataset)

Most ML tutorials use a dataset that's already clean and a problem that's
already labeled. Real fraud/AML work almost never looks like that — you
rarely have a labeled "this was fraud" column, and you have to *manufacture*
your own validation signal. This tutorial is deliberately built around that:
in stage 1 you plant known patterns into your own synthetic data specifically
so that in stages 2-3 you can measure whether your own detection logic
actually catches them. That loop — plant it, try to catch it, measure how
well you did — is the actual daily rhythm of this kind of work.
