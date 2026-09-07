# Stage 1: Synthetic Data

## The concept

You can't practice on your employer's real customer data (confidential,
and honestly you shouldn't want to). So the first skill is: **generate
fake data that behaves like the real thing.**

The trap most people fall into: generating data that's *purely random*.
If every transaction is a random amount at a random time, there's nothing
to detect — "detecting" random noise is meaningless. Real fraud detection
works because bad behavior has **patterns**: someone structuring payments
keeps every transaction just under a reporting threshold; someone laundering
money moves it fast through many small transactions; watchlist evasion
produces names that are *close to* but not *identical to* a flagged name.

So in this stage you'll generate three tables, and deliberately **plant**
three known patterns into the transaction data — so that later stages have
something real to find, and you have a way to check whether your detection
code actually works.

### The three tables

- **`watchlist.csv`** — a small list of "flagged" entities (sanctions/PEP-style)
- **`customers.csv`** — your bank's customers, where a handful are secretly
  near-duplicate names of watchlist entities (simulating aliases/misspellings)
- **`transactions.csv`** — mostly normal transactions, plus three planted
  patterns:
  1. **Structuring**: several transactions just under a $10,000 threshold,
     clustered in a day or two
  2. **Velocity**: many transactions in a very short time window
  3. **High-risk corridor**: large transactions to/from flagged countries

## What you'll build

Open `starter/generate_data.py`. It has four functions with `# TODO`s:

- `make_name_variant(name)` — given a name, return a plausible "alias":
  a misspelling, dropped letter, or reordering. This is what makes stage 2
  interesting — exact string matching won't catch these.
- `generate_watchlist(n)` — build a small DataFrame of fake flagged entities
- `generate_customers(watchlist_df, n)` — build customers, secretly making
  ~15 of them near-duplicates of watchlist names (keep track of which ones —
  you'll need this "ground truth" later to check your own accuracy)
- `generate_transactions(customers_df)` — build mostly-normal transactions,
  then add the three planted patterns described above

Use the `faker` library for realistic-looking names/dates
(`from faker import Faker; fake = Faker()`).

**Important design choice:** add a column `is_planted_alias_for_testing`
to your customers table (True/False). This is *not* something a real
system would have — it's your own answer key, so that in Stage 2 you can
measure precision/recall instead of eyeballing results.

## Checkpoint

Run this from the **project root** (not inside the `starter/` folder) —
every stage shares one `data/` folder so later stages can read what
earlier stages wrote:

```bash
python3 01-synthetic-data/starter/generate_data.py
```

You should see output like:

```
Watchlist:     25 entities
Customers:     600 (15 planted watchlist-alias matches)
Transactions:  ~4000+
```

And three CSVs written to a `data/` folder. Open `transactions.csv` and
spot-check: can you find a customer with several transactions between
$8,500-$9,950 within a day or two of each other? That's your planted
structuring pattern.

## Stuck?

Compare against `../../solutions/src/generate_data.py`. Don't copy-paste —
read it, understand the *why* behind each choice, then go back and finish
yours.

## Next

→ [Stage 2: Entity Resolution](../02-entity-resolution/README.md)
