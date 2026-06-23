"""
FinBERT fine-tuning with MLflow experiment tracking.
"""

import json
import logging
import mlflow
import mlflow.pytorch
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def patch_model_config(model_name):
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    if "id2label" in config and isinstance(next(iter(config["id2label"].values())), float):
        config["id2label"] = {k: f"LABEL_{k}" for k in config["id2label"]}
        config["label2id"] = {v: int(k) for k, v in config["id2label"].items()}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def load_data(params):
    cfg = params["dataset"]

    patch_model_config(params["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])

    dataset = load_dataset(cfg["name"], split="train")

    dataset = dataset.shuffle(seed=cfg["seed"]).select(range(cfg["num_samples"]))

    raw_labels = dataset[cfg["label_column"]]
    # Dataset uses -1/0/1; model expects 0/1/2 (negative/neutral/positive).
    unique_labels = sorted(set(raw_labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    dataset = dataset.map(
        lambda ex: {cfg["label_column"]: label2id[ex[cfg["label_column"]]]}
    )

    def tokenize(batch):
        return tokenizer(
            batch[cfg["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=params["model"]["max_length"],
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column(cfg["label_column"], "labels")

    # Drop all unnecessary columns
    keep = ["input_ids", "attention_mask", "labels"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    dataset.set_format("torch")

    split = dataset.train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    return split["train"], split["test"], num_labels


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
    train_ds, val_ds, num_labels = load_data(params)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        params["model"]["name"],
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    with mlflow.start_run():
        # Log all hyperparameters once at the start of the run.
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

        # Add tags for filtering
        for key, value in mf["tags"].items():
            mlflow.set_tag(key, value)

        print("Training...")
        for epoch in range(tr["num_epochs"]):
            loss = train_epoch(model, train_loader, optimizer, device, tr["max_steps"])
            acc = evaluate(model, val_loader, device)

            # Log metrics at each epoch.
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)

            print(
                f"Epoch {epoch + 1}/{tr['num_epochs']}  loss={loss:.4f}  val_acc={acc:.4f}"
            )
        print("Training complete.")

        # Save the trained model as an MLflow artifact
        # This takes some time
        mlflow.pytorch.log_model(model, name=mf["model_artifact_name"], serialization_format="pickle")
        print(f"Run complete. Model logged as artifact '{mf['model_artifact_name']}'.")


if __name__ == "__main__":
    main()
