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
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # should print: Using device: cuda

print("Loading dataset...")
# data = fetch_diabetes_hospital(as_frame=True)
data = fetch_diabetes_hospital(as_frame=True, cache=True)
X, y = data.data, data.target

# TODO: Extract the sensitive feature (gender)
sensitive_feature = X["gender"].copy()
# TODO: drop protected attributes (gender, race) and the target column (readmitted) from X
X = X.drop(columns=["gender", "race", "readmitted"], errors="ignore")

# TODO: Encode all non-numeric columns using LabelEncoder
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
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
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)

class MLP(nn.Module):
    def __init__(self, input_dim=24, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
    def forward(self, x):
        return self.net(x)

print("Training MLP...")
# model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
model = NeuralNetClassifier(
    MLP,
    module__input_dim=X_train.shape[1],
    module__output_dim=len(np.unique(y_train)),
    max_epochs=100,
    lr=0.001,
    batch_size=256,         # larger batch = better GPU utilization
    device=device,          # ← this line is what actually puts the model on the GPU/CPU
    train_split=None,       # mirrors original — no internal val split
    verbose=1,
)
model.fit(X_train, y_train)

joblib.dump({"model": model, "scaler": scaler}, "model.joblib")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
sf_test.reset_index(drop=True).to_csv("sf_test.csv", index=False)

print("Saved: model.joblib, X_test.npy, y_test.npy, sf_test.csv")
