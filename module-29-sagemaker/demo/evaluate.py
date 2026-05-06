"""
Evaluation script for SageMaker Processing.

Inputs (injected by SageMaker):
  /opt/ml/processing/model  — trained model artifact (model.tar.gz)
  /opt/ml/processing/test   — test.json from preprocessing step

Output:
  /opt/ml/processing/evaluation/evaluation.json
"""

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "transformers>=4.26.0", "datasets>=2.14.0"],
    check=True,
)

import json
import os
import tarfile

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_input_dir = "/opt/ml/processing/model"
test_path       = "/opt/ml/processing/test"
output_dir      = "/opt/ml/processing/evaluation"

print("Extracting model artifact...")
tar_path = os.path.join(model_input_dir, "model.tar.gz")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(model_input_dir)

print("Loading model and tokenizer...")
model     = AutoModelForSequenceClassification.from_pretrained(model_input_dir)
tokenizer = AutoTokenizer.from_pretrained(model_input_dir)

print("Loading test data...")
test_df      = pd.read_json(f"{test_path}/test.json", lines=True)
test_dataset = Dataset.from_pandas(test_df)

def tokenize(example):
    return tokenizer(example["Title"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

predictions, true_labels = [], []
model.eval()

with torch.no_grad():
    for item in test_dataset:
        outputs = model(
            input_ids=item["input_ids"].unsqueeze(0),
            attention_mask=item["attention_mask"].unsqueeze(0),
        )
        predictions.append(torch.argmax(outputs.logits, dim=1).item())
        true_labels.append(item["label"].item())

accuracy = float(accuracy_score(true_labels, predictions))
print(f"\nAccuracy: {accuracy:.4f}")
print(classification_report(true_labels, predictions))

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump({"metrics": {"accuracy": {"value": accuracy}}}, f, indent=2)

print(f"Evaluation saved to {output_dir}/evaluation.json")
