"""
Retrieves online features for a given cardholder and measures latency
against a direct Parquet file lookup.
"""

import time

import pandas as pd
from feast import FeatureStore

CARDHOLDER_ID = 42
FEATURES = [
    "transaction_stats:transaction_amount",
    "transaction_stats:transaction_frequency",
    "transaction_stats:average_spend",
    "behavioral_features:days_since_last_transaction",
    "behavioral_features:transaction_velocity",
]

if __name__ == "__main__":
    store = FeatureStore(repo_path=".")

    # TODO: Call store.get_online_features() with the feature list and a single
    # entity row for CARDHOLDER_ID, then convert the result to a dict
    print(f"Retrieving online features for cardholder_id={CARDHOLDER_ID}...")
    start = time.perf_counter()
    feature_vector = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"cardholder_id": CARDHOLDER_ID}],
    ).to_dict()
    online_latency = (time.perf_counter() - start) * 1000

    print("\nFeature vector from online store:")
    for key, values in feature_vector.items():
        print(f"  {key}: {values[0]}")

    # TODO: Read transactions.parquet, filter to CARDHOLDER_ID, sort by
    # event_timestamp, and take the most recent row as the baseline comparison
    start = time.perf_counter()
    df = pd.read_parquet("data/transactions.parquet")
    result = df[df["cardholder_id"] == CARDHOLDER_ID].sort_values("event_timestamp").iloc[-1]
    parquet_latency = (time.perf_counter() - start) * 1000

    print(f"\nLatest values from Parquet:")
    print(f"  transaction_amount         : {result['transaction_amount']}")
    print(f"  transaction_frequency      : {result['transaction_frequency']}")
    print(f"  average_spend              : {result['average_spend']}")
    print(f"  days_since_last_transaction: {result['days_since_last_transaction']}")
    print(f"  transaction_velocity       : {result['transaction_velocity']}")


    print(f"\nLatency comparison:")
    print(f"  Online store (Feast SQLite) : {online_latency:.2f} ms")
    print(f"  Parquet file scan           : {parquet_latency:.2f} ms")
    print(f"  Speedup                     : {parquet_latency / online_latency:.1f}x")
