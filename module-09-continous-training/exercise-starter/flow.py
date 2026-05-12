"""
Continuous training flow
"""

import shutil

import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# TODO Create a train task using train.py
def train_model() -> str:
    pass


# TODO Create a validation task using validate.py
def validate_model() -> bool:
    pass


def save_model() -> None:
    """Copies the candidate model to the final output path."""
    params = load_params()
    src = params["paths"]["model_candidate"]
    dst = params["paths"]["model_output"]
    shutil.copy2(src, dst)
    print(f"Model saved to {dst}.")


# TODO Define the continous training worklow using the tasks above
def continuous_training_flow():
    pass


if __name__ == "__main__":
    # TODO Create a deployment using serve() that listens to the
    # "new-data-available" event
