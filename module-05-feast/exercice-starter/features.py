"""
Feast feature definitions.

Entity: cardholder_id

Feature views:
  - transaction_stats     : amount, frequency, average_spend
  - behavioral_features   : days_since_last_transaction, transaction_velocity
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

TRANSACTIONS_PATH = str(Path(__file__).parent / "data" / "transactions.parquet")
ACTIVITY_PATH = str(Path(__file__).parent / "data" / "cardholder_activity.parquet")

# TODO: Define the cardholder entity with name "cardholder_id"
# Read more https://docs.feast.dev/getting-started/concepts/entity
cardholder = ...

# TODO: Define a FileSource with name "transaction_source" pointing to transactions.parquet,
# using "event_timestamp" as the timestamp field
transaction_source = ...

# TODO: Define a FileSource with name "activity_source" pointing to cardholder_activity.parquet,
# using "event_timestamp" as the timestamp field
activity_source = ...

# TODO: Define the transaction_stats FeatureView with fields:
# transaction_amount (Float64), transaction_frequency (Int64), average_spend (Float64)
# ttl=30 days, linked to the cardholder entity and transaction_source
# Read more https://docs.feast.dev/getting-started/concepts/feature-view
transaction_stats = ...

# TODO: Define the behavioral_features FeatureView with fields:
# days_since_last_transaction (Int64), transaction_velocity (Float64)
# ttl=30 days, linked to the cardholder entity and activity_source
behavioral_features = ...