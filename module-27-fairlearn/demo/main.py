"""
Evaluating model fairness with Fairlearn.
Run:
    python main.py
"""

import json

import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data 

print("Loading diabetes hospital dataset...")
data = fetch_diabetes_hospital(as_frame=True)
X, y = data.data, data.target

# Sensitive attribute: race
sensitive_col = "race"
sensitive_feature = X[sensitive_col].copy()

X = X.drop(columns=[sensitive_col, "readmitted"], errors="ignore")

# Encode categoricals
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.astype(float)

y = LabelEncoder().fit_transform(y)


X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLP

print("Training MLP classifier...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nOverall accuracy: {overall_acc:.4f}")

# Fairness evaluation

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

print("\nPer-subgroup metrics (race):")
print(mf.by_group.to_string())

print(f"\nOverall metrics:")
print(mf.overall.to_string())

print(f"\nGroup differences (max - min):")
print(mf.difference().to_string())

dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)
print(f"\nDemographic parity difference : {dpd:.4f}")
print(f"Equalized odds difference     : {eod:.4f}")


report = {
    "sensitive_attribute": sensitive_col,
    "overall_accuracy": overall_acc,
    "demographic_parity_difference": dpd,
    "equalized_odds_difference": eod,
    "by_group": mf.by_group.round(4).to_dict(),
}

with open("fairness_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to fairness_report.json")
