"""
Compares the latest trained model against the best model seen so far.
Returns True if the new model is better and should be saved, False otherwise.
"""

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_runs(client, experiment_name):
    # TODO fetch all runs for the experiment,
    # ordered by start_time (most recent first).
    # Raise a RuntimeError if no runs are found.
    # Return the list of runs.
    pass


def main():
    params = load_params()
    mf = params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    runs = get_runs(client, mf["experiment_name"])

    # TODO Get the val_accuracy of the most recent run (runs[0]).

    
    # TODO Find the best val_accuracy across all previous runs.
    # Hint: use max() with a generator expression.
    # Hint: Consider the case where there is only one run.

    # TODO Return True if the new model strictly improves on the best
    #      previous accuracy, False otherwise
    pass


if __name__ == "__main__":
    main()
