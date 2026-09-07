"""
generate_data.py
----------------
Generates synthetic customers, a sanctions/watchlist, and transactions
with deliberately planted AML red flags, including a common-name
collision test case (same name, different person, different DOB).
"""

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 600
N_WATCHLIST = 25
N_NORMAL_TXNS = 4000
STRUCTURING_THRESHOLD = 10_000
HIGH_RISK_COUNTRIES = ["Country_X", "Country_Y", "Country_Z"]
NORMAL_COUNTRIES = ["USA", "Canada", "UK", "Germany", "India", "Australia", "France", "Japan"]


def make_name_variant(name: str) -> str:
    variants = [
        lambda n: n.replace("a", "e", 1) if "a" in n else n + "e",
        lambda n: n.replace(" ", "-"),
        lambda n: n[:-1] if len(n) > 3 else n,
        lambda n: n.replace("ph", "f") if "ph" in n else n,
        lambda n: n.replace("i", "y", 1) if "i" in n else n,
        lambda n: n + " Jr",
        lambda n: n.split()[0] + " " + n.split()[-1][0] + "." if len(n.split()) > 1 else n,
    ]
    fn = random.choice(variants)
    try:
        return fn(name)
    except Exception:
        return name


def generate_watchlist(n=N_WATCHLIST) -> pd.DataFrame:
    rows = []
    for i in range(n):
        name = fake.name()
        rows.append({
            "watchlist_id": f"WL-{i:04d}",
            "entity_name": name,
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80),
            "list_source": random.choice(["OFAC-STYLE-SIM", "PEP-STYLE-SIM", "INTERNAL-CASE-SIM"]),
            "risk_category": random.choice(["Sanctions", "PEP", "Adverse Media"]),
        })
    return pd.DataFrame(rows)


def generate_customers(watchlist_df: pd.DataFrame, n=N_CUSTOMERS) -> pd.DataFrame:
    rows = []
    n_planted = 15          # true positives: aliased name + SAME date of birth
    n_common_name = 10      # false-positive test: EXACT name match, DIFFERENT person (different DOB)

    planted_source_rows = watchlist_df.sample(n_planted, random_state=1)
    common_name_source_rows = watchlist_df.sample(n_common_name, random_state=2)

    for i in range(n):
        if i < n_planted:
            src = planted_source_rows.iloc[i]
            name = make_name_variant(src["entity_name"])
            dob = src["date_of_birth"]  # same person -> DOB genuinely matches
            is_planted_alias = True
            is_common_name_collision = False
        elif i < n_planted + n_common_name:
            src = common_name_source_rows.iloc[i - n_planted]
            name = src["entity_name"]  # EXACT name match, but a different, unrelated person
            dob = fake.date_of_birth(minimum_age=18, maximum_age=80)  # different DOB
            is_planted_alias = False
            is_common_name_collision = True
        else:
            name = fake.name()
            dob = fake.date_of_birth(minimum_age=18, maximum_age=80)
            is_planted_alias = False
            is_common_name_collision = False

        rows.append({
            "customer_id": f"CUST-{i:05d}",
            "customer_name": name,
            "date_of_birth": dob,
            "country": random.choice(NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES),
            "account_open_date": fake.date_between(start_date="-5y", end_date="-30d"),
            "is_planted_alias_for_testing": is_planted_alias,
            "is_common_name_collision_for_testing": is_common_name_collision,
        })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def generate_transactions(customers_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cust_ids = customers_df["customer_id"].tolist()
    cust_country = dict(zip(customers_df["customer_id"], customers_df["country"]))

    start_date = datetime(2025, 1, 1)
    for _ in range(N_NORMAL_TXNS):
        cid = random.choice(cust_ids)
        rows.append({
            "txn_id": str(uuid.uuid4())[:8],
            "customer_id": cid,
            "amount": round(np.random.lognormal(mean=6.5, sigma=1.0), 2),
            "timestamp": start_date + timedelta(days=random.randint(0, 240), minutes=random.randint(0, 1440)),
            "counterparty_country": cust_country[cid] if random.random() > 0.2
                else random.choice(NORMAL_COUNTRIES + HIGH_RISK_COUNTRIES),
            "channel": random.choice(["wire", "ach", "card", "cash"]),
        })

    structuring_customers = random.sample(cust_ids, 12)
    for cid in structuring_customers:
        base_day = random.randint(0, 220)
        for _ in range(random.randint(4, 7)):
            rows.append({
                "txn_id": str(uuid.uuid4())[:8],
                "customer_id": cid,
                "amount": round(random.uniform(8500, 9950), 2),
                "timestamp": start_date + timedelta(days=base_day + random.randint(0, 2), minutes=random.randint(0, 1440)),
                "counterparty_country": cust_country[cid],
                "channel": "cash",
            })

    velocity_customers = random.sample([c for c in cust_ids if c not in structuring_customers], 10)
    for cid in velocity_customers:
        base_day = random.randint(0, 220)
        for _ in range(random.randint(8, 15)):
            rows.append({
                "txn_id": str(uuid.uuid4())[:8],
                "customer_id": cid,
                "amount": round(random.uniform(500, 5000), 2),
                "timestamp": start_date + timedelta(days=base_day, minutes=random.randint(0, 180)),
                "counterparty_country": cust_country[cid],
                "channel": random.choice(["wire", "ach"]),
            })

    corridor_customers = random.sample(cust_ids, 8)
    for cid in corridor_customers:
        rows.append({
            "txn_id": str(uuid.uuid4())[:8],
            "customer_id": cid,
            "amount": round(random.uniform(15000, 60000), 2),
            "timestamp": start_date + timedelta(days=random.randint(0, 240)),
            "counterparty_country": random.choice(HIGH_RISK_COUNTRIES),
            "channel": "wire",
        })

    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    watchlist_df = generate_watchlist()
    customers_df = generate_customers(watchlist_df)
    transactions_df = generate_transactions(customers_df)

    import os
    os.makedirs("data", exist_ok=True)
    watchlist_df.to_csv("data/watchlist.csv", index=False)
    customers_df.to_csv("data/customers.csv", index=False)
    transactions_df.to_csv("data/transactions.csv", index=False)

    print(f"Watchlist:     {len(watchlist_df)} entities")
    print(f"Customers:     {len(customers_df)} "
          f"({customers_df['is_planted_alias_for_testing'].sum()} planted aliases, "
          f"{customers_df['is_common_name_collision_for_testing'].sum()} common-name collisions)")
    print(f"Transactions:  {len(transactions_df)}")


if __name__ == "__main__":
    main()