"""
Preprocessing script for SageMaker Processing.

Downloads the financial headlines dataset from HuggingFace, encodes labels
(Global Sentiment -1/0/1 → 0/1/2), and splits into train / valid / test JSON files.

Output paths (SageMaker Processing output):
  /opt/ml/processing/output/train/train.json
  /opt/ml/processing/output/valid/valid.json
  /opt/ml/processing/output/test/test.json
"""

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "numpy>=1.24.0,<2.0.0",
    "pyarrow>=14.0.0,<16.0.0",
    "datasets>=2.14.0,<3.0.0",
], check=True)

import os

from datasets import load_dataset

OUTPUT_BASE = "/opt/ml/processing/output"

print("Downloading dataset: baptle/financial_headlines_market_based...")
dataset = load_dataset("baptle/financial_headlines_market_based", split="train")

# Encode -1/0/1 sentiment → 0/1/2 (required by model)
unique_labels = sorted(set(dataset["Global Sentiment"]))
label2id = {lbl: i for i, lbl in enumerate(unique_labels)}

def encode(example):
    example["label"] = label2id[example["Global Sentiment"]]
    return example

dataset = dataset.map(encode).remove_columns(["Global Sentiment"])
print(f"Total samples: {len(dataset)}  |  Classes: {label2id}")

# 80 / 10 / 10 split
train_test  = dataset.train_test_split(test_size=0.2, seed=42)
valid_test  = train_test["test"].train_test_split(test_size=0.5, seed=42)

splits = {
    "train": train_test["train"],
    "valid": valid_test["train"],
    "test":  valid_test["test"],
}

for name, split in splits.items():
    out_dir = os.path.join(OUTPUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)
    split.to_json(os.path.join(out_dir, f"{name}.json"))
    print(f"Saved {name}: {len(split)} rows → {out_dir}/{name}.json")

print("Preprocessing complete.")
