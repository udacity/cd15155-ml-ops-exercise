"""
Evaluates the trained MLP classifier for fairness across gender and race subgroups.

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
sf_test = pd.read_csv("sf_test.csv")
sf_gender = sf_test["gender"]
sf_race = sf_test["race"]

y_pred = model.predict(X_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_acc:.4f}")

# TODO: Build a MetricFrame with accuracy, precision, and recall broken down by
# the sensitive features (gender and race) and print per-subgroup results
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

print(f"\nPer-subgroup metrics (gender x race):")
print(mf.by_group.to_string())

# TODO: Compute demographic_parity_difference and equalized_odds_difference
# for both gender and race so we can compare the gaps
dpd_gender = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_gender)
eod_gender = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_gender)
dpd_race = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_race)
eod_race = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_race)

print(f"\nDemographic parity difference (gender) : {dpd_gender:.4f}  (threshold: {DPD_THRESHOLD})")
print(f"Equalized odds difference     (gender) : {eod_gender:.4f}  (threshold: {EOD_THRESHOLD})")
print(f"Demographic parity difference (race)   : {dpd_race:.4f}  (threshold: {DPD_THRESHOLD})")
print(f"Equalized odds difference     (race)   : {eod_race:.4f}  (threshold: {EOD_THRESHOLD})")

# TODO: Save a fairness report with the per-group metrics and gap comparisons to disk
report = {
    "sensitive_attributes": ["gender", "race"],
    "overall_accuracy": overall_acc,
    "demographic_parity_difference_gender": dpd_gender,
    "equalized_odds_difference_gender": eod_gender,
    "demographic_parity_difference_race": dpd_race,
    "equalized_odds_difference_race": eod_race,
    "by_group": mf.by_group.round(4).reset_index().to_dict(orient="records"),
}
with open("fairness_report.json", "w") as f:
    json.dump(report, f, indent=2)

# TODO: Implement the fairness gate. Exit with code 1 if any metric exceeds its threshold.
metric_names = np.array([
    "demographic_parity_difference (gender)",
    "equalized_odds_difference (gender)",
    "demographic_parity_difference (race)",
    "equalized_odds_difference (race)",
])
values = np.array([dpd_gender, eod_gender, dpd_race, eod_race])
thresholds = np.array([DPD_THRESHOLD, EOD_THRESHOLD, DPD_THRESHOLD, EOD_THRESHOLD])
exceeded = values > thresholds

if exceeded.any():
    print("\nFairness check FAILED:")
    for name, value, threshold in zip(metric_names[exceeded], values[exceeded], thresholds[exceeded]):
        print(f"  - {name} {value:.4f} > {threshold}")
    sys.exit(1)

print("\nFairness check passed.")
