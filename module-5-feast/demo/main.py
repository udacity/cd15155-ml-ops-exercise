"""
Demo: Feast feature store

Run:
    python generate_data.py
    feast apply
    python main.py
"""

import subprocess
import pandas as pd
from datetime import datetime, timezone
from feast import FeatureStore


def main():
    store = FeatureStore(repo_path=".")

    # Entity dataframe: cardholder IDs, the timestamp of each label event,
    # and a fraud label. Feast will join features available BEFORE each timestamp.
    entity_df = pd.DataFrame({
        "cardholder_id": [1, 2, 3, 4, 5],
        "event_timestamp": [
            datetime(2024, 1, 10, tzinfo=timezone.utc),
            datetime(2024, 1, 16, tzinfo=timezone.utc),
            datetime(2024, 1, 22, tzinfo=timezone.utc),
            datetime(2024, 1, 28, tzinfo=timezone.utc),
            datetime(2024, 2,  3, tzinfo=timezone.utc),
        ],
        "is_fraud": [0, 1, 0, 1, 0],
    })

    print("Entity dataframe:")
    print(entity_df.to_string(index=False))

    print("\nFetching historical features with point-in-time correctness...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_stats:transaction_amount",
            "transaction_stats:transaction_frequency",
            "transaction_stats:average_spend",
        ],
    ).to_df()

    print("\nTraining dataset (features joined at label timestamp):")
    print(training_df.to_string(index=False))


if __name__ == "__main__":
    main()
