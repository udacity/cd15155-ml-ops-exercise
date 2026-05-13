"""
Data Drift for textual data with Evidently AI.
"""

import json
import sys

import pandas as pd


# NOTE: Download the datasets using the download_data script
REFERENCE_PATH = "data/reviews_2026_feb_nyc.csv"
CURRENT_PATH = "data/reviews_2026_feb_albany.csv"

# TODO: Define a sample size, keep it small since encoding is compute-intensive
SAMPLE_SIZE = ...

# TODO: Define the drift threshold for the embedding model-based test
EMBEDDING_THRESHOLD = ...


# TODO: Load reviews CSV and return a clean sample of comments (drop NAs)
def load_reviews(path: str, n: int) -> list[str]:
    pass


print("Loading reviews...")
ref_comments = load_reviews(REFERENCE_PATH, SAMPLE_SIZE)
cur_comments = load_reviews(CURRENT_PATH, SAMPLE_SIZE)

print(f"Reference (NYC)   : {len(ref_comments)} reviews")
print(f"Current   (Albany): {len(cur_comments)} reviews")

# TODO: Encode reviews using the sentence-transformers library. 
# Hint: The comments are multilingual 
# Check out the available models here: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-similarity-models
print("\nEncoding reviews with paraphrase-multilingual-MiniLM-L12-v2...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# TODO: Encode both review lists into DataFrames and rename columns to "col_0", "col_1", ...
# Evidently requires named columns to identify the embedding dimensions.
ref_embeddings = ...
ref_embeddings.columns = ...

cur_embeddings = ...
cur_embeddings.columns = ...


print("Running embedding drift report...")
# TODO: Create a ColumnMapping that groups all embedding columns under a single
# name ("review_embeddings"). Evidently needs this to know that col_0,
# col_1, ... col_383 are not independent features but one embedding vector
# Hint: use the embeddings parameter of ColumnMapping
# Read more about ColumnMapping here: https://github.com/evidentlyai/evidently/blob/a4aa4c2b37fe7a4344cc5031f566deccf3d69e4f/src/evidently/legacy/pipeline/column_mapping.py#L28
column_mapping = ...

# TODO: Create a report with EmbeddingsDriftMetric and the model-based drift method
# https://github.com/evidentlyai/evidently/blob/a4aa4c2b37fe7a4344cc5031f566deccf3d69e4f/src/evidently/legacy/metrics/data_drift/embeddings_drift.py#L51

report = ...

# TODO: Run the report


# TODO: Save HTML report


# TODO: Implement the drift gate logic by extracting the drift score and drift detected flag from the report result.

