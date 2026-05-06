"""
Demo: Model Quality Testing with Deepchecks

Checks:
  1. TrainTestPerformance    — overall F1 on train vs test
  2. WeakSegmentsPerformance — subgroups where the model struggles
  3. ConfusionMatrixReport   — false positives vs false negatives
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from deepchecks.core import CheckFailure
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ConfusionMatrixReport,
    TrainTestPerformance,
    WeakSegmentsPerformance,
)

class SpamMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def load_data():
    ds = load_dataset("nahiar/facebook_spam_detection", split="train")
    df = ds.to_pandas()

    label_col = "Label"
    drop_cols = [label_col, "profile id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Sanitize column names for Deepchecks compatibility
    clean_names = {
        c: c.replace("#", "num_").replace("/", "_per_").replace(" ", "_")
        for c in feature_cols
    }
    df = df.rename(columns=clean_names)
    feature_cols = list(clean_names.values())

    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"Dataset shape: {df.shape}  NaN remaining: {df[feature_cols].isna().sum().sum()}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_cols


def train(model: SpamMLP, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50):
    # Weighted loss to handle class imbalance (83% legit / 17% spam)
    # pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
    criterion = nn.BCEWithLogitsLoss() #pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={epoch_loss / len(loader):.4f}")

def run_inference(model: SpamMLP, X: np.ndarray):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        proba = torch.sigmoid(logits).numpy()
    preds = (proba >= 0.5).astype(int)
    proba_2d = np.column_stack([1 - proba, proba])
    return preds, proba_2d


def build_datasets(X_train, X_test, y_train, y_test, feature_cols):
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["label"] = y_train.astype(int)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["label"] = y_test.astype(int)

    train_ds = Dataset(train_df, label="label", cat_features=[])
    test_ds = Dataset(test_df, label="label", cat_features=[])
    return train_ds, test_ds


def build_suite():
    return Suite(
        "Spam Detection Quality Suite",
        TrainTestPerformance().add_condition_test_performance_greater_than(0.80),
        WeakSegmentsPerformance(),
        ConfusionMatrixReport(),
    )



def main():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"Train: {len(X_train)} samples  Test: {len(X_test)} samples")
    print(f"Spam rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    print("\nTraining PyTorch MLP...")
    model = SpamMLP(input_dim=X_train.shape[1])
    train(model, X_train, y_train, epochs=50)

    print("\nRunning inference...")
    train_preds, train_proba = run_inference(model, X_train)
    test_preds, test_proba = run_inference(model, X_test)
    print(f"Train accuracy: {(train_preds == y_train.astype(int)).mean():.4f}")
    print(f"Test  accuracy: {(test_preds  == y_test.astype(int)).mean():.4f}")

    print("\nPreparing Deepchecks datasets...")
    train_ds, test_ds = build_datasets(X_train, X_test, y_train, y_test, feature_cols)

    print("Running quality suite...")
    suite = build_suite()
    result = suite.run(
        train_ds, test_ds,
        y_pred_train=train_preds,
        y_pred_test=test_preds,
        y_proba_train=train_proba,
        y_proba_test=test_proba,
    )

    with open("report.json", "w") as f:
        f.write(result.to_json())
    print("\nReport saved to report.json")

    print("\nChecks summary:")
    for check_result in result.results:
        if isinstance(check_result, CheckFailure):
            print(f"  [ERROR] {check_result.check.name()}: {check_result.exception}")
        elif check_result.have_conditions():
            status = "PASS" if check_result.passed_conditions() else "FAIL"
            print(f"  [{status}] {check_result.check.name()}")
        else:
            print(f"  [INFO] {check_result.check.name()}")


if __name__ == "__main__":
    main()
