"""
Compares the latest run (challenger) against the best previous run (champion).
Returns True if the challenger should replace the champion, False otherwise.
"""

import logging
import mlflow
import yaml

logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_runs(client, experiment_name):
    # TODO fetch all runs for the experiment,
    # ordered by start_time (most recent first).
    # Raise a RuntimeError if no runs are found.
    # Return the list of runs.
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if not runs:
        raise RuntimeError("No runs found in the experiment.")
    return runs


def main():
    params = load_params()
    mf = params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    runs = get_runs(client, mf["experiment_name"])

    # TODO Get the val_accuracy of the most recent run.
    challenger_accuracy = runs[0].data.metrics["val_accuracy"]

    if len(runs) == 1:
        print(f"Challenger accuracy : {challenger_accuracy:.4f}")
        print("First run. Model promoted by default.")
        return True

    # TODO Find the best val_accuracy across all previous runs.
    # Hint: use max() with a generator expression.
    # Hint: Consider the case where there is only one run.    
    champion_accuracy = max(r.data.metrics["val_accuracy"] for r in runs[1:])

    print(f"Challenger accuracy : {challenger_accuracy:.4f}")
    print(f"Champion accuracy   : {champion_accuracy:.4f}")

    # TODO Return True if the new model strictly improves on the best
    # previous accuracy, False otherwise.
    if challenger_accuracy > champion_accuracy:
        print("Model will be saved.")
        return True

    print("Accuracy didn't improve. Model will not be saved.")
    return False


if __name__ == "__main__":
    main()
