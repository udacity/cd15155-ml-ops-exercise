"""
Generates a point-in-time correct training dataset using Feast.

Run after: feast apply
"""

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore

if __name__ == "__main__":
    store = FeatureStore(repo_path=".")

    # TODO: Build an entity DataFrame with cardholder IDs, label timestamps,
    # and fraud labels.
    from datetime import datetime, timezone
    now = datetime.now(tz=timezone.utc)
    entity_df = pd.DataFrame({
        "cardholder_id": list(range(1, 21)),
        "event_timestamp": [now] * 20,
        "is_fraud": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    })

    # TODO: Call store.get_historical_features() with the entity DataFrame and
    # the list of features from both feature views, then convert to a DataFrame
    print("Fetching historical features with point-in-time correctness...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:transaction_frequency",
            "transaction_stats:average_spend",
            "behavioral_features:days_since_last_transaction",
            "behavioral_features:transaction_velocity",
        ],
    ).to_df()

    print(f"\nTraining dataset shape: {training_df.shape}")
    print(training_df.head(10).to_string(index=False))

    # TODO: Save the training dataset to a parquet file
    training_df.to_parquet("data/training_dataset.parquet", index=False)
    print("\nSaved to data/training_dataset.parquet")
