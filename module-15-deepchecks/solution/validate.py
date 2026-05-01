"""
Validates the @dev model before promoting it to @production.

Step 1 — Champion/challenger: compare @dev accuracy against @production.
Step 2 — Quality gate: run a Deepchecks vision suite on the @dev model predictions.

Returns True only if both steps pass.
"""

import mlflow
import numpy as np
import yaml
from datasets import load_dataset
from deepchecks.core import CheckFailure
from deepchecks.vision import VisionData, Suite
from deepchecks.vision.vision_data import BatchOutputFormat
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from deepchecks.vision.suites import train_test_validation


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def get_accuracy(client, model_name, alias):
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)
    return run.data.metrics["val_accuracy"], version.version


def champion_challenger(client, model_name):
    """Returns True if @dev beats @production (or no production exists yet)."""
    dev_accuracy, dev_version = get_accuracy(client, model_name, "dev")

    try:
        prod_accuracy, prod_version = get_accuracy(client, model_name, "production")
    except Exception:
        print(f"@dev     : v{dev_version}  accuracy={dev_accuracy:.4f}")
        print("No @production model found. @dev will go straight to quality checks.")
        return True

    print(f"@dev        : v{dev_version}  accuracy={dev_accuracy:.4f}")
    print(f"@production : v{prod_version}  accuracy={prod_accuracy:.4f}")

    if dev_accuracy > prod_accuracy:
        print("@dev beats @production...running quality checks.")
        return True

    print("@dev did not improve over @production.")
    return False

class BeansDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

#TODO Create a collate function that will be use in the DataLoader
# The collate function that transforms a batch into the format expected by Deepchecks
# The function takes a batch as input and returns a batch of images, labels and predictions for the batch
# Read more here: https://docs.deepchecks.com/stable/vision/auto_tutorials/quickstarts/plot_classification_tutorial.html
def make_collate_fn(pipe, label_names):
    """Returns a collate function that runs the pipeline and formats for Deepchecks."""
    def collate_fn(examples):
        images_pil = [ex["image"].convert("RGB") for ex in examples]
        labels = [ex["labels"] for ex in examples]

        # Run HuggingFace pipeline on the batch
        predictions_raw = pipe(images_pil)
        proba = []
        for preds in predictions_raw:
            scores = {p["label"]: p["score"] for p in preds}
            proba.append([scores.get(lbl, 0.0) for lbl in label_names])

        # Convert PIL images to numpy (H, W, C) uint8 for Deepchecks
        images_np = [np.array(img) for img in images_pil]

        return BatchOutputFormat(
            images=images_np,
            labels=labels,
            predictions=proba,
        )
    return collate_fn

#TODO Create a function that create a VisionData object
# Use the collate function above to a dataloader in deepchecks format
# and then use the dataloader to create a VisionData object that will be used in the Deepchecks suite
def build_vision_data(hf_split, pipe, label_names, batch_size=8):
    dataset = BeansDataset(load_dataset("beans", split=hf_split))
    label_map = {i: lbl for i, lbl in enumerate(label_names)}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(pipe, label_names),
    )
    return VisionData(batch_loader=loader, task_type="classification", label_map=label_map)

#TODO Implement model/data quality checks using Deepchecks
def run_quality_checks(pipe, label_names):
    """Builds VisionData loaders, runs the Deepchecks suite, returns True if all checks pass."""


    print("Building train VisionData...")
    train_data = build_vision_data("train", pipe, label_names)

    print("Building test VisionData...")
    test_data = build_vision_data("test", pipe, label_names)


    print("Running Deepchecks quality suite...")
    #TODO build deepcheck's train test validation suite and run it on the train/test vision datasets
    suite = train_test_validation()
    result = suite.run(train_data, test_data, max_samples=5000)

    failed = []
    print("\nChecks summary:")
    for check_result in result.results:
        if isinstance(check_result, CheckFailure):
            print(f"  [ERROR] {check_result.check.name()}: {check_result.exception}")
            continue

        check_name = check_result.check.name()
        check_failed = any(not c.is_pass() for c in check_result.conditions_results)

        if check_result.conditions_results:
            print(f"  [{'PASS' if not check_failed else 'FAIL'}] {check_name}")
            if check_failed:
                failed.append(check_name)
        else:
            print(f"  [INFO] {check_name}")

    return len(failed) == 0


def main():
    params = load_params()
    mf = params["mlflow"]
    model_name = params["registry"]["model_name"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    # Step 1: champion / challenger
    if not champion_challenger(client, model_name):
        return False

    # Step 2: load @dev model and run quality checks
    print("\nLoading @dev model from MLflow registry...")
    pipe = mlflow.transformers.load_model(f"models:/{model_name}@dev")

    dataset = load_dataset("beans", split="validation")
    label_names = dataset.features["labels"].names

    return run_quality_checks(pipe, label_names)


if __name__ == "__main__":
    passed = main()
    print("\nValidation passed — ready to promote." if passed else "\nValidation FAILED — not promoting.")
