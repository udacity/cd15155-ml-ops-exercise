"""
Fairness gate 
loads a pre-trained model and evaluates it.
Run:
    python check_fairness.py
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# TODO: Define thresholds for demographic parity difference and equalized odds difference
DPD_THRESHOLD = 0.10
EOD_THRESHOLD = 0.10

# TODO: Load the pre-trained model and test data saved by train.py
# check train.py for reference on what files to load and their formats
print("Loading model and test data...")
artifacts = joblib.load("model.joblib")
model = artifacts["model"]
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
sf_test = pd.read_csv("sf_test.csv").squeeze()

y_pred = model.predict(X_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_acc:.4f}")

# TODO: Build a MetricFrame with accuracy, precision, and recall broken down by
# the sensitive feature (gender) and print per-subgroup results
metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sf_test,
)

print(f"\nPer-subgroup metrics (gender):")
print(mf.by_group.to_string())

# TODO: Compute demographic_parity_difference and equalized_odds_difference
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)

print(f"\nDemographic parity difference : {dpd:.4f}  (threshold: {DPD_THRESHOLD})")
print(f"Equalized odds difference     : {eod:.4f}  (threshold: {EOD_THRESHOLD})")

# Export JSON report

report = {
    "sensitive_attribute": "gender",
    "overall_accuracy": overall_acc,
    "demographic_parity_difference": dpd,
    "equalized_odds_difference": eod,
    "thresholds": {
        "demographic_parity_difference": DPD_THRESHOLD,
        "equalized_odds_difference": EOD_THRESHOLD,
    },
    "by_group": mf.by_group.round(4).to_dict(),
}

with open("fairness_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to fairness_report.json")

# TODO: Implement the fairness gate — exit with code 1 if either metric exceeds its threshold
failed = []
if dpd > DPD_THRESHOLD:
    failed.append(f"demographic_parity_difference {dpd:.4f} > {DPD_THRESHOLD}")
if eod > EOD_THRESHOLD:
    failed.append(f"equalized_odds_difference {eod:.4f} > {EOD_THRESHOLD}")

if failed:
    print("\nFairness check FAILED:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)

print("\nFairness check passed.")
