import os

import pandas as pd
import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def preprocess(raw_path, processed_path):
    df = pd.read_csv(raw_path, parse_dates=["time"])

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values")

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved {len(df)} processed rows to {processed_path}")


if __name__ == "__main__":
    params = load_params()
    preprocess(params["paths"]["raw"], params["paths"]["processed"])
