"""
Train the MLP classifier and save the model + test data to disk.
Run:
    python train.py
"""

import joblib
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Loading dataset...")
data = fetch_diabetes_hospital(as_frame=True)
X, y = data.data, data.target

# TODO: Extract the sensitive feature (gender)
sensitive_feature = ...
# TODO: drop protected attributes (gender, race) and the target column (readmitted) from X
X = ...

# TODO: Encode all non-numeric columns using LabelEncoder


X = X.astype(float)

# TODO: Encode target labels
y = LabelEncoder().fit_transform(y)

# TODO: split into train/test sets, keeping the sensitive feature aligned with the split
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training MLP...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump({"model": model, "scaler": scaler}, "model.joblib")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
sf_test.reset_index(drop=True).to_csv("sf_test.csv", index=False)

print("Saved: model.joblib, X_test.npy, y_test.npy, sf_test.csv")
