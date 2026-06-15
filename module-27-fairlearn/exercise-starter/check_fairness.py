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
DPD_THRESHOLD = ...
EOD_THRESHOLD = ...

# TODO: Load the pre-trained model and test data saved by train.py
# check train.py for reference on what files to load and their formats
print("Loading model and test data...")

model = ...
X_test = ...
y_test = ...
sf_test = ...
# TODO: Extract the gender and race columns from sf_test
sf_gender = ...
sf_race = ...

y_pred = model.predict(X_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_acc:.4f}")

# TODO: Build a MetricFrame with accuracy, precision, and recall broken down by
# the sensitive features (gender and race) and print per-subgroup results
# https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html#fairlearn.metrics.MetricFrame
metrics = ...

mf = ...

print(f"\nPer-subgroup metrics (gender x race):")
print(mf.by_group.to_string())

# TODO: Compute demographic_parity_difference and equalized_odds_difference
# for both gender and race so we can compare the gaps
# https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
dpd_gender = ...
eod_gender = ...
dpd_race = ...
eod_race = ...

print(f"\nDemographic parity difference (gender) : {dpd_gender:.4f}  (threshold: {DPD_THRESHOLD})")
print(f"Equalized odds difference     (gender) : {eod_gender:.4f}  (threshold: {EOD_THRESHOLD})")
print(f"Demographic parity difference (race)   : {dpd_race:.4f}  (threshold: {DPD_THRESHOLD})")
print(f"Equalized odds difference     (race)   : {eod_race:.4f}  (threshold: {EOD_THRESHOLD})")

# TODO: Save a fairness report with the per-group metrics and gap comparisons to disk

# TODO: Implement the fairness gate. Exit with code 1 if any metric exceeds its threshold.

