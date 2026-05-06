"""
Data Drift for textual data with Evidently AI.
"""

import json
import sys

import pandas as pd
from sentence_transformers import SentenceTransformer

from evidently.legacy.metrics import EmbeddingsDriftMetric
from evidently.legacy.metrics.data_drift.embedding_drift_methods import model
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

#NOTE Download the datasets using the download_data script
REFERENCE_PATH = "data/reviews_2026_feb_nyc.csv"
CURRENT_PATH = "data/reviews_2026_feb_albany.csv"

# TODO: Define sample size, keep small since encoding is compute-intensive
SAMPLE_SIZE = 500

# TODO: Define the drift threshold for the embedding model-based test
EMBEDDING_THRESHOLD = 0.55


# TODO: Load reviews CSV and return a clean sample of comments
def load_reviews(path: str, n: int) -> list[str]:
    df = pd.read_csv(path)
    return (
        df["comments"]
        .dropna()
        .sample(n, random_state=42)
        .reset_index(drop=True)
        .tolist()
    )


print("Loading reviews...")
n = min(SAMPLE_SIZE, 500)
ref_comments = load_reviews(REFERENCE_PATH, n)
cur_comments = load_reviews(CURRENT_PATH, n)

print(f"Reference (NYC)   : {len(ref_comments)} reviews")
print(f"Current   (Albany): {len(cur_comments)} reviews")

# TODO: Encode reviews using an embedding model
# Hint: The comments are multilingual
print("\nEncoding reviews with paraphrase-multilingual-MiniLM-L12-v2...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

ref_embeddings = pd.DataFrame(encoder.encode(ref_comments))
ref_embeddings.columns = ["col_" + str(c) for c in ref_embeddings.columns]

cur_embeddings = pd.DataFrame(encoder.encode(cur_comments))
cur_embeddings.columns = ["col_" + str(c) for c in cur_embeddings.columns]


print("Running embedding drift report...")
# TODO: Create a ColumnMapping that groups all embedding columns under a single
# name ("review_embeddings"). Evidently needs this to know that col_0,
# col_1, ... col_383 are not independent features but one embedding vector —
# only then will it run EmbeddingsDriftMetric instead of per-column drift.
column_mapping = ColumnMapping(
    embeddings={"review_embeddings": list(ref_embeddings.columns)}
)
# TODO: Run EmbeddingsDriftMetric with the model-based drift method
report = Report(metrics=[
    EmbeddingsDriftMetric(
        "review_embeddings",
        drift_method=model(
            threshold=EMBEDDING_THRESHOLD,
            bootstrap=None,
            quantile_probability=0.95,
            pca_components=None,
        ),
    )
])

report.run(
    reference_data=ref_embeddings,
    current_data=cur_embeddings,
    column_mapping=column_mapping,
)

result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["drift_detected"]
drift_score = round(result["metrics"][0]["result"]["drift_score"], 4)

print(f"\nDrift detected : {drift_detected}")
print(f"Drift score    : {drift_score}  (threshold: {EMBEDDING_THRESHOLD})")

# TODO: Save HTML report
report.save_html("drift_report_text.html")
print("\nSaved: drift_report_text.html")

# TODO: Implement the drift gate
if drift_detected:
    print(f"\nDrift gate FAILED. Semantic drift detected between NYC and Albany reviews")
    sys.exit(1)

print("\nDrift gate passed.")
