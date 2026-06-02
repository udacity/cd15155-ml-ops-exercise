"""
Generates a point-in-time correct training dataset using Feast.
"""

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore

if __name__ == "__main__":
    store = FeatureStore(repo_path=".")

    # Show raw data before any feature engineering
    raw_df = pd.read_parquet("data/transactions.parquet")
    print("----- Raw Transactions Data -----")
    print(f"Shape: {raw_df.shape}")
    print(raw_df.head(5).to_string(index=False))

    from datetime import datetime, timezone
    now = datetime.now(tz=timezone.utc)
    entity_df = pd.DataFrame({
        "cardholder_id": list(range(1, 21)),
        "event_timestamp": [now] * 20,
        "is_fraud": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    })
    print("\n----- Entity DataFrame (query) -----")
    print(f"Shape: {entity_df.shape}")
    print(entity_df.head(5).to_string(index=False))

    # TODO: Retrieve historical features with the entity DataFrame and
    # the list of features from both feature views
    # then convert to a DataFrame
    # Read more: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
    print("\nFetching historical features with point-in-time correctness...")
    training_df = ...

    print("\n=== Feature Dataset (after point-in-time join) ===")
    print(f"Shape: {training_df.shape}")
    print(training_df.head(5).to_string(index=False))

    # TODO: Save the training dataset to a parquet file

