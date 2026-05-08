"""
Validates the latest MLflow run against a minimum accuracy threshold.
Returns True if the model passes, False otherwise.
"""

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_latest_run_accuracy(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in the experiment.")
    return runs[0].data.metrics["val_accuracy"]


def main():
    params = load_params()
    mf = params["mlflow"]
    threshold = params["validation"]["accuracy_threshold"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    accuracy = get_latest_run_accuracy(client, mf["experiment_name"])

    print(f"val_accuracy : {accuracy:.4f}")
    print(f"threshold    : {threshold}")

    if accuracy >= threshold:
        print("PASSED. model meets the quality threshold.")
        return True

    print(f"FAILED. {accuracy:.4f} is below threshold {threshold}.")
    return False


if __name__ == "__main__":
    main()
