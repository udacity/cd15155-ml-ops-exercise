"""
Validates the @dev model before promoting it to @production.

Step 1: Champion/challenger: compare @dev accuracy against @production.
Step 2: Quality gate: run a Deepchecks vision suite on the @dev model predictions.

Returns True only if both steps pass.
"""

import mlflow
import numpy as np
import yaml
from datasets import load_dataset
import logging
from mlflow.tracking import MlflowClient
logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


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
# The collate function transforms a batch into the format expected by Deepchecks
# The function takes a batch as input and returns a batch of images, labels and predictions for the batch
# Read more here: https://docs.deepchecks.com/stable/vision/auto_tutorials/quickstarts/plot_classification_tutorial.html#implementing-the-visiondata-class
def make_collate_fn(pipe, label_names):
    """Returns a collate function that runs the pipeline and formats for Deepchecks."""
    def collate_fn(examples):
        #TODO Convert each example's image to RGB and collect the labels
        images_pil = ...
        labels = ...

        #TODO Run the HuggingFace pipeline on the batch to get the predictions probabilities
        predictions_raw = ...
        proba = ...

        #TODO Convert PIL images to numpy (H, W, C) uint8 for Deepchecks
        images_np = ...

        #TODO Return a BatchOutputFormat with the images, labels and predictions
        raise NotImplementedError("TODO: implement the collate function")
    return collate_fn

#TODO Create a function that create a VisionData object
# Use the collate function above to create a dataloader in deepchecks format
# and then use the dataloader to create a VisionData object that will be used in the Deepchecks suite
def build_vision_data(hf_split, pipe, label_names, dataset_name, batch_size=8):
    dataset = ...
    label_map = ...
    loader = ...
    vision_data = ...
    return vision_data

#TODO Implement model quality checks using Deepchecks
def run_quality_checks(pipe, label_names, dataset_name):

    #TODO Build Deepchecks VisionData objects for the train and test sets using the build_vision_data function above
    # Hint: https://docs.deepchecks.com/stable/vision/usage_guides/visiondata_object.html
    print("Building train VisionData...")
    train_data = ...

    print("Building test VisionData...")
    test_data = ...


    print("Running Deepchecks quality suite...")
    #TODO build deepcheck's model evaluation suite and run it on the train/test vision datasets
    #https://docs.deepchecks.com/stable/api/generated/deepchecks.vision.suites.model_evaluation.html#deepchecks.vision.suites.model_evaluation
    suite = ...
    result = ...

    result.save_as_html("output.html", as_widget=False)
    print("Report saved to output.html")
    
    #TODO Return False if any of the suite's conditions failed, otherwise return True
    raise NotImplementedError("TODO: implement the quality checks and return True/False based on the results")


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

    dataset_name = params["dataset"]["name"]
    dataset = load_dataset(dataset_name, split="validation")
    label_names = dataset.features["labels"].names

    return run_quality_checks(pipe, label_names, dataset_name)


if __name__ == "__main__":
    passed = main()
    print("\nValidation passed. Ready to promote." if passed else "\nValidation FAILED...Not promoting.")
