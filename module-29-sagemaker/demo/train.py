"""
Training script for SageMaker HuggingFace estimator.

Dataset : baptle/financial_headlines_market_based (preprocessed by preprocess.py)
Model   : baptle/FinBERT_market_based
Task    : Financial headline sentiment classification (3 classes)

SageMaker injects:
  SM_CHANNEL_TRAIN  — path to train.json
  SM_CHANNEL_VALID  — path to valid.json
  SM_MODEL_DIR      — where to save the trained model
"""

import os

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "baptle/FinBERT_market_based"


def main():
    train_path = os.environ["SM_CHANNEL_TRAIN"]
    valid_path = os.environ["SM_CHANNEL_VALID"]
    model_dir  = os.environ["SM_MODEL_DIR"]

    print(f"Loading train data from {train_path}...")
    train_df = pd.read_json(f"{train_path}/train.json", lines=True)
    valid_df = pd.read_json(f"{valid_path}/valid.json", lines=True)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(example["Title"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    valid_dataset = valid_dataset.rename_column("label", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="/opt/ml/output/logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
