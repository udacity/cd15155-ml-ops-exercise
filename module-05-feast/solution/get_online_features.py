"""
Retrieves online features for a given cardholder
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

    # TODO: Retrieve online features for the specified cardholder_id and features list
    # Read more: https://docs.feast.dev/master/getting-started/concepts/feature-retrieval
    
    print(f"Retrieving online features for cardholder_id={CARDHOLDER_ID}...")

    feature_vector = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"cardholder_id": CARDHOLDER_ID}],
    ).to_dict()

    print("\nFeature vector from online store:")
    for key, values in feature_vector.items():
        print(f"  {key}: {values[0]}")

    # TODO: Read both parquet files, filter to CARDHOLDER_ID, sort by
    # event_timestamp, and take the most recent row as the baseline comparison
    transactions = pd.read_parquet("data/transactions.parquet")
    activity = pd.read_parquet("data/cardholder_activity.parquet")
    t_row = transactions[transactions["cardholder_id"] == CARDHOLDER_ID].sort_values("event_timestamp").iloc[-1]
    a_row = activity[activity["cardholder_id"] == CARDHOLDER_ID].sort_values("event_timestamp").iloc[-1]

    print(f"\nLatest values from Parquet files:")
    print(f"  transaction_amount              : {t_row['transaction_amount']}")
    print(f"  transaction_frequency           : {t_row['transaction_frequency']}")
    print(f"  average_spend                   : {t_row['average_spend']}")
    print(f"  days_since_last_transaction     : {a_row['days_since_last_transaction']}")
    print(f"  transaction_velocity            : {a_row['transaction_velocity']}")