"""
Continuous training flow.

Run to register the deployment with the local Prefect server:
    python flow.py

The deployment listens for "new-data-available" events emitted by add_data.py
and triggers a retrain automatically.
"""

import shutil

import yaml
from prefect import flow, task
from prefect.events import DeploymentEventTrigger


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


@task(name="Train", log_prints=True)
def train_model() -> str:
    import train
    return train.main()


@task(name="Validate", log_prints=True)
def validate_model() -> bool:
    import validate
    return validate.main()


@task(name="Save Model", log_prints=True)
def save_model() -> None:
    params = load_params()
    src = params["paths"]["model_candidate"]
    dst = params["paths"]["model_output"]
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    print(f"Model saved to {dst}.")


@flow(name="continuous-training-flow", log_prints=True)
def continuous_training_flow():
    train_model()
    passed = validate_model()
    if passed:
        save_model()
    else:
        print("Model failed validation. Model not saved.")


if __name__ == "__main__":
    continuous_training_flow.serve(
        name="local",
        triggers=[
            DeploymentEventTrigger(
                enabled=True,
                match={"prefect.resource.id": "training-data"},
                expect=["new-data-available"],
            )
        ],
    )
