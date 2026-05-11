"""
Retrieves online features for a given cardholder and measures latency
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
    # measure the latency of the online feature retrieval
    # Read more: https://docs.feast.dev/master/getting-started/concepts/feature-retrieval
    


    # TODO: Read transactions.parquet, filter to CARDHOLDER_ID, sort by
    # event_timestamp, and take the most recent row as the baseline comparison
