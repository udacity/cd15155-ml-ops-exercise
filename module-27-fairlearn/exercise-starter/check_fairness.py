"""
Evaluates the trained MLP classifier for fairness across gender subgroups.

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

y_pred = model.predict(X_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_acc:.4f}")

# TODO: Build a MetricFrame with accuracy, precision, and recall broken down by
# the sensitive feature (gender) and print per-subgroup results
# https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html#fairlearn.metrics.MetricFrame
metrics = ...

mf = ...


# TODO: Compute demographic_parity_difference and equalized_odds_difference
# https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
dpd = ...
eod = ...

# TODO: Implement the fairness gate. Exit with code 1 if either metric exceeds its threshold.

