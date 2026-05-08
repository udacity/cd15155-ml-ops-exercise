import os

import pandas as pd
import requests
import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def fetch_weather(api_params):
    url = "https://archive-api.open-meteo.com/v1/archive"
    response = requests.get(
        url,
        params={
            "latitude": api_params["latitude"],
            "longitude": api_params["longitude"],
            "start_date": api_params["start_date"],
            "end_date": api_params["end_date"],
            "hourly": ",".join(api_params["variables"]),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def save_raw(data, path):
    df = pd.DataFrame(data["hourly"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    params = load_params()
    print("Fetching weather data from Open-Meteo...")
    data = fetch_weather(params["api"])
    save_raw(data, params["paths"]["raw"])
