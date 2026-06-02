"""
Loads the Production model from the MLflow registry
and runs inference on a sample image from the beans dataset.

Usage:
    python predict.py
    python predict.py --image path/to/image.jpg
"""

import argparse

import mlflow
import yaml

# TODO Import MlflowClient
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# TODO load the production model from MLflow registry
# Hint: https://mlflow.org/docs/latest/ml/model-registry/tutorial/#example-2-load-via-model-version-alias
def load_production_model(tracking_uri: str, model_name: str):
    #TODO set the tracking URI for MLflow

    #TODO Get model information (version) using the model version alias

    version = ...
 

    #TODO Get the model URI
    model_uri = ...
    #TODO Load the model using the model URI
    print(f"Loading model from: {model_uri}  (version {version})")
    return None


# TODO run inference using the production model and return the predicted label
def predict(model, image):
    raise NotImplementedError("Implement predict() function")


def main(image_path=None):
    params = load_params()
    mf = params["mlflow"]
    model_name = params["registry"]["model_name"]

    print(f"Loading Production model '{model_name}' from registry...")

    model = load_production_model(mf["tracking_uri"], model_name)

    if image_path is None:
        from datasets import load_dataset

        dataset = load_dataset(
            params["dataset"]["name"], split=params["dataset"]["split"]
        )
        sample = dataset[0]
        image = sample["image"]
        true_label = dataset.features["labels"].names[sample["labels"]]
        print(f"Using sample image. True label: {true_label}")
    else:
        from PIL import Image

        image = Image.open(image_path)
        true_label = "unknown"

    # TODO Call predict() and print the result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()
    main(args.image)
