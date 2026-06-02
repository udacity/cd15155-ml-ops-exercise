"""
Continuous training flow

Run to register the deployment with the local Prefect server:
    python flow.py
"""

import shutil

import yaml
from prefect import flow, task
from prefect.events import DeploymentEventTrigger


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# TODO Create a train task using train.py
@task(name="Train", log_prints=True)
def train_model() -> str:
    import train

    return train.main()


# TODO Create a validation task using validate.py
@task(name="Validate", log_prints=True)
def validate_model() -> bool:
    import validate

    return validate.main()


@task(name="Save Model", log_prints=True)
def save_model() -> None:
    params = load_params()
    src = params["paths"]["model_candidate"]
    dst = params["paths"]["model_output"]
    shutil.copy2(src, dst)
    print(f"Model saved to {dst}.")


# TODO Define the continous training worklow using the tasks above
@flow(name="continuous-training-flow", log_prints=True)
def continuous_training_flow():
    train_model()
    passed = validate_model()
    if passed:
        save_model()
    else:
        print("Model failed validation. Skipping saving.")


if __name__ == "__main__":
    # TODO Create a deployment using serve() that listens to the
    # "new-data-available"
    continuous_training_flow.serve(
        name="local",
        triggers=[
            DeploymentEventTrigger(
                enabled=True,
                match={"prefect.resource.id": "sales-training-data"},
                expect=["new-data-available"],
            )
        ],
    )
