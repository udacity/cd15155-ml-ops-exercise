"""
Feast feature definitions 
Entity: cardholder_id
Feature view: transaction_stats
  - transaction_amount
  - transaction_frequency
  - average_spend
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

DATA_PATH = str(Path(__file__).parent / "data" / "transactions.parquet")

cardholder = Entity(
    name="cardholder_id",
    description="Unique identifier for each cardholder",
)

transaction_source = FileSource(
    path=DATA_PATH,
    timestamp_field="event_timestamp",
)

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
    description="Transaction statistics per cardholder",
)
