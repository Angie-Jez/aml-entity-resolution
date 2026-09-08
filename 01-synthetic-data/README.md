Stage 1: Synthetic Data — Done

I couldn't use my employer's real customer data (confidential, and I shouldn't want to anyway), so I started by generating fake data that behaves like the real thing.

The trap I made sure to avoid: generating data that's purely random. If every transaction were a random amount at a random time, there'd be nothing to detect — "detecting" random noise is meaningless. Real fraud detection works because bad behavior has patterns: someone structuring payments keeps every transaction just under a reporting threshold; someone laundering money moves it fast through many small transactions; watchlist evasion produces names that are close to but not identical to a flagged name. So I deliberately planted three known patterns into my transaction data, giving later stages something real to find and giving myself a way to check whether my detection code actually works.

The three tables I built
watchlist.csv — a small list of "flagged" entities (sanctions/PEP-style)
customers.csv — my bank's customers, where I secretly made a handful near-duplicate names of watchlist entities (simulating aliases/misspellings)
transactions.csv — mostly normal transactions, plus three planted patterns:
Structuring: several transactions just under a $10,000 threshold, clustered in a day or two
Velocity: many transactions in a very short time window
High-risk corridor: large transactions to/from flagged countries
What I built

I opened starter/generate_data.py and filled in its four # TODO functions:

make_name_variant(name) — given a name, returns a plausible "alias": a misspelling, dropped letter, or reordering. This is what makes Stage 2 interesting — exact string matching won't catch these.
generate_watchlist(n) — builds a small DataFrame of fake flagged entities.
generate_customers(watchlist_df, n) — builds customers, secretly making ~15 of them near-duplicates of watchlist names. I kept track of which ones — that "ground truth" is what I'll need later to check my own accuracy.
generate_transactions(customers_df) — builds mostly-normal transactions, then adds the three planted patterns above.

I used the faker library for realistic-looking names/dates (from faker import Faker; fake = Faker()).

Design choice I made: I added a column is_planted_alias_for_testing to my customers table (True/False). This isn't something a real system would have — it's my own answer key, so in Stage 2 I can measure precision/recall instead of eyeballing results.

Checkpoint

I ran this from the project root (not inside starter/), since every stage shares one data/ folder so later stages can read what earlier stages wrote:

bash
python3 01-synthetic-data/starter/generate_data.py

And got output like:

Watchlist:     25 entities
Customers:     600 (15 planted watchlist-alias matches)
Transactions:  ~4000+

Three CSVs were written to the data/ folder. I opened transactions.csv and spot-checked it — I found a customer with several transactions between $8,500–$9,950 within a day or two of each other. That's my planted structuring pattern, confirmed.
