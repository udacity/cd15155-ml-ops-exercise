"""
Generates a synthetic credit card transaction feature dataset and saves it
as a Parquet file for use with Feast.

Run:
    python generate_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

NUM_CARDHOLDERS = 100
TIMESTAMPS_PER_CARDHOLDER = 10
SEED = 42

np.random.seed(SEED)

rows = []
from datetime import timezone
base_time = datetime.now(tz=timezone.utc) - timedelta(days=TIMESTAMPS_PER_CARDHOLDER * 3)

for cardholder_id in range(1, NUM_CARDHOLDERS + 1):
    for i in range(TIMESTAMPS_PER_CARDHOLDER):
        event_timestamp = base_time + timedelta(days=i * 3)
        rows.append({
            "cardholder_id": cardholder_id,
            "event_timestamp": event_timestamp,
            "transaction_amount": round(np.random.exponential(scale=150), 2),
            "transaction_frequency": int(np.random.poisson(lam=5)),
            "average_spend": round(np.random.normal(loc=200, scale=80), 2),
        })

df = pd.DataFrame(rows)
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

os.makedirs("data", exist_ok=True)
df.to_parquet("data/transactions.parquet", index=False)
print(f"Saved {len(df)} rows to data/transactions.parquet")
print(df.head())
