"""
FinBERT fine-tuning with MLflow experiment tracking.
"""

import mlflow
import mlflow.pytorch
import torch
import yaml
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_data(params):
    cfg = params["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])

    dataset = load_dataset(cfg["name"], split="train")

    dataset = dataset.shuffle(seed=cfg["seed"]).select(range(cfg["num_samples"]))

    raw_labels = dataset[cfg["label_column"]]
    if isinstance(raw_labels[0], str):
        label2id = {l: i for i, l in enumerate(sorted(set(raw_labels)))}
        dataset = dataset.map(lambda ex: {"label": label2id[ex[cfg["label_column"]]]})

    def tokenize(example):
        return tokenizer(
            example[cfg["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=params["model"]["max_length"],
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column(cfg["label_column"], "labels")
    dataset.set_format("torch")

    split = dataset.train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    return split["train"], split["test"]


def train_epoch(model, loader, optimizer, device, max_steps):
    model.train()
    total_loss, steps = 0.0, 0
    for batch in loader:
        if max_steps > 0 and steps >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return correct / total


def main():
    params = load_params()
    tr = params["training"]
    mf = params["mlflow"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading and tokenizing dataset...")
    train_ds, val_ds = load_data(params)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"])

    print("Loading model...")
    num_labels = AutoConfig.from_pretrained(params["model"]["name"]).num_labels
    model = AutoModelForSequenceClassification.from_pretrained(
        params["model"]["name"],
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])

    # Point MLflow at the local tracking server started with `mlflow ui`.
    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    with mlflow.start_run():
        # Log all hyperparameters once at the start of the run.
        # These appear in the MLflow UI under the "Parameters" tab.
        mlflow.log_params(
            {
                "learning_rate": tr["learning_rate"],
                "batch_size": tr["batch_size"],
                "num_epochs": tr["num_epochs"],
                "max_length": params["model"]["max_length"],
                "num_samples": params["dataset"]["num_samples"],
                "model": params["model"]["name"],
            }
        )

        # Tags are free-form key-value pairs — useful for filtering runs
        # in the UI (e.g. find all runs tagged stage=dev).
        for key, value in mf["tags"].items():
            mlflow.set_tag(key, value)

        print("Training...")
        for epoch in range(tr["num_epochs"]):
            loss = train_epoch(model, train_loader, optimizer, device, tr["max_steps"])
            acc = evaluate(model, val_loader, device)

            # Log a metric at each epoch. The `step` argument is what
            # produces the x-axis in the MLflow metrics chart.
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)

            print(
                f"Epoch {epoch + 1}/{tr['num_epochs']}  loss={loss:.4f}  val_acc={acc:.4f}"
            )

        # Save the trained model as an MLflow artifact linked to this run.
        # This is what enables model registry and deployment later.
        mlflow.pytorch.log_model(model, mf["model_artifact_name"])
        print(f"Run complete. Model logged as artifact '{mf['model_artifact_name']}'.")


if __name__ == "__main__":
    main()
