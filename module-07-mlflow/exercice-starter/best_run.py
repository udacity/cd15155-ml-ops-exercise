"""
Retrieve the best run from the MLflow experiment and display its parameters
and metrics using the MLflow client.
"""

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_best_run(client, experiment_name, metric="val_accuracy"):
    # TODO: Look up the experiment by name. Raise a ValueError if it does not exist.
    # Search all its runs sorted by the given metric from highest to lowest.
    # Raise a ValueError if no runs exist yet.
    # Return the run with the best metric value.
    experiment = ...

    runs = ...

    return runs[0]


def display_run(run, metric="val_accuracy"):
    print(f"Best run ID : {run.info.run_id}")
    print(f"Status      : {run.info.status}")
    print()

    print("Parameters:")
    for key, value in run.data.params.items():
        print(f"  {key:<20} {value}")

    print()
    print("Metrics:")
    for key, value in run.data.metrics.items():
        print(f"  {key:<20} {value:.4f}")

    print()
    print("Tags:")
    for key, value in run.data.tags.items():
        if not key.startswith("mlflow."):
            print(f"  {key:<20} {value}")


def main():
    params = load_params()
    mf = params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    print(f"Querying experiment: '{mf['experiment_name']}'")
    best = get_best_run(client, mf["experiment_name"])
    display_run(best)


if __name__ == "__main__":
    main()
