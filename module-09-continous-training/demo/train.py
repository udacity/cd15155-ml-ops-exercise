"""
FinBERT fine-tuning with MLflow tracking.
"""

import mlflow
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_data(params):
    cfg = params["dataset"]
    df = pd.read_csv("data/train.csv")
    df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    raw_labels = df[cfg["label_column"]].tolist()
    unique_labels = sorted(set(raw_labels))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels = torch.tensor([label2id[lbl] for lbl in raw_labels], dtype=torch.long)
    texts = df[cfg["text_column"]].tolist()

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=params["model"]["max_length"],
        return_tensors="pt",
    )

    split = int(len(labels) * (1 - cfg["val_split"]))
    train_ds = TensorDataset(
        encodings["input_ids"][:split],
        encodings["attention_mask"][:split],
        labels[:split],
    )
    val_ds = TensorDataset(
        encodings["input_ids"][split:],
        encodings["attention_mask"][split:],
        labels[split:],
    )
    return train_ds, val_ds, len(unique_labels)


def train_epoch(model, loader, optimizer, device, max_steps):
    model.train()
    total_loss, steps = 0.0, 0
    for input_ids, attention_mask, labels in loader:
        if max_steps > 0 and steps >= max_steps:
            break
        loss = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            preds = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def main():
    params = load_params()
    tr = params["training"]
    mf = params["mlflow"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data from data/train.csv...")
    train_ds, val_ds, num_labels = load_data(params)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"])
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        params["model"]["name"],
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "learning_rate": tr["learning_rate"],
                "batch_size": tr["batch_size"],
                "num_epochs": tr["num_epochs"],
                "max_length": params["model"]["max_length"],
                "train_samples": len(train_ds),
                "model": params["model"]["name"],
            }
        )
        for key, value in mf["tags"].items():
            mlflow.set_tag(key, value)

        print("Training...")
        for epoch in range(tr["num_epochs"]):
            loss = train_epoch(model, train_loader, optimizer, device, tr["max_steps"])
            acc = evaluate(model, val_loader, device)
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            print(
                f"Epoch {epoch + 1}/{tr['num_epochs']}  loss={loss:.4f}  val_acc={acc:.4f}"
            )

        model.save_pretrained(params["paths"]["model_candidate"])
        run_id = run.info.run_id
        print(
            f"Run complete. Candidate saved to {params['paths']['model_candidate']}  run_id={run_id}"
        )
        return run_id


if __name__ == "__main__":
    main()
