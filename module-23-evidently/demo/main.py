"""
Data drift detection with Evidently AI.
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/listings_2025_march.csv"
CURRENT_PATH = "data/listings_2025_november.csv"

FEATURES = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "room_type",
    "neighbourhood_group_cleansed",
]

DRIFT_THRESHOLD = 0.1
SAMPLE_SIZE = 5000
MAX_PRICE = 1000


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[FEATURES].copy()
    df["price"] = (
        df["price"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
    )
    df = df.dropna()
    # Drop extreme price outliers so the drift charts show the actual distribution
    return df[df["price"] <= MAX_PRICE]


print("Loading data...")
df_ref = load_and_clean(REFERENCE_PATH)
df_cur = load_and_clean(CURRENT_PATH)

n = min(SAMPLE_SIZE, len(df_ref), len(df_cur))
df_ref = df_ref.sample(n, random_state=42).reset_index(drop=True)
df_cur = df_cur.sample(n, random_state=42).reset_index(drop=True)

print(f"Reference (2025 March)   : {len(df_ref)} rows")
print(f"Current   (2025 November): {len(df_cur)} rows")

print("\nRunning drift report...")
report = Report([DataDriftPreset(threshold=DRIFT_THRESHOLD)])
snapshot = report.run(reference_data=df_ref, current_data=df_cur)

result = snapshot.dict()

drifted_count_metric = next(
    m for m in result["metrics"] if "DriftedColumnsCount" in m["metric_name"]
)
n_drifted = int(drifted_count_metric["value"]["count"])
print(f"\nDrifted features : {n_drifted} / {len(FEATURES)}")
print("Open drift_report.html for full visualizations.")

snapshot.save_html("drift_report.html")
print("Report saved to drift_report.html")
