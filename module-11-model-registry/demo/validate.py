"""
Compares the @dev model version against the @production version.
Returns True if the dev version should be promoted, False otherwise.
"""

import mlflow
import yaml
import logging
from mlflow.tracking import MlflowClient
logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_accuracy(client, model_name, alias):
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)
    return run.data.metrics["val_accuracy"], version.version


def main():
    params = load_params()
    mf = params["mlflow"]
    model_name = params["registry"]["model_name"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    dev_accuracy, dev_version = get_accuracy(client, model_name, "dev")

    try:
        prod_accuracy, prod_version = get_accuracy(client, model_name, "production")
    except Exception:
        print(f"@dev     : v{dev_version}  accuracy={dev_accuracy:.4f}")
        print("No @production alias found. dev version will be promoted.")
        return True

    print(f"@dev        : v{dev_version}  accuracy={dev_accuracy:.4f}")
    print(f"@production : v{prod_version}  accuracy={prod_accuracy:.4f}")

    if dev_accuracy > prod_accuracy:
        print("dev beats production. Model will be promoted.")
        return True

    print("No improvement over production.")
    return False


if __name__ == "__main__":
    main()
