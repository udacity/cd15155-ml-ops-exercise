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

DATA_PATH = str(Path(__file__).parent / "data" / "transactions.parquet")

# TODO: Define the cardholder entity with name "cardholder_id"
cardholder = Entity(
    name="cardholder_id",
    description="Unique identifier for each cardholder",
)

# TODO: Define a FileSource pointing to transactions.parquet,
# using "event_timestamp" as the timestamp field
transaction_source = FileSource(
    path=DATA_PATH,
    timestamp_field="event_timestamp",
)

# TODO: Define the transaction_stats FeatureView with fields:
# transaction_amount (Float64), transaction_frequency (Int64), average_spend (Float64)
# ttl=30 days, linked to the cardholder entity and transaction_source
transaction_stats = FeatureView(
    name="transaction_stats",
    entities=[cardholder],
    ttl=timedelta(days=30),
    schema=[
        Field(name="transaction_amount", dtype=Float64),
        Field(name="transaction_frequency", dtype=Int64),
        Field(name="average_spend", dtype=Float64),
    ],
    source=transaction_source,
    description="Aggregated transaction statistics per cardholder",
)

# TODO: Define the behavioral_features FeatureView with fields:
# days_since_last_transaction (Int64), transaction_velocity (Float64)
# ttl=30 days, linked to the cardholder entity and transaction_source
behavioral_features = FeatureView(
    name="behavioral_features",
    entities=[cardholder],
    ttl=timedelta(days=30),
    schema=[
        Field(name="days_since_last_transaction", dtype=Int64),
        Field(name="transaction_velocity", dtype=Float64),
    ],
    source=transaction_source,
    description="Behavioral signals per cardholder",
)
