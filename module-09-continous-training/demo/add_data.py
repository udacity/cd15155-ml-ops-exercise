"""
Simulates a stream of new data arriving over time.
Each call appends the next chunk to data/train.csv.

Usage:
    python add_data.py            # 5 chunks by default
    python add_data.py --chunks 3
"""

import argparse
import os

import pandas as pd
import yaml
from datasets import load_dataset

CHUNKS_DIR = "data/chunks"
TRAIN_CSV = "data/train.csv"
STATE_FILE = "data/.chunk_state"


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def read_state():
    if not os.path.exists(STATE_FILE):
        return 0
    with open(STATE_FILE) as f:
        return int(f.read().strip())


def write_state(chunk_index):
    with open(STATE_FILE, "w") as f:
        f.write(str(chunk_index))


def setup_chunks(params, n_chunks):
    cfg = params["dataset"]
    print(f"Downloading {cfg['name']}...")
    dataset = load_dataset(cfg["name"], split="train")
    df = (
        dataset.to_pandas()
        .sample(frac=1, random_state=cfg["seed"])
        .reset_index(drop=True)
    )
    print(f"Full dataset: {len(df)} rows")

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunk_size = len(df) // n_chunks
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(df)
        df.iloc[start:end].to_csv(f"{CHUNKS_DIR}/chunk_{i + 1}.csv", index=False)
        print(f"  chunk_{i + 1}.csv has {end - start} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=5)
    args = parser.parse_args()

    params = load_params()
    current = read_state()

    if current == 0:
        setup_chunks(params, args.chunks)
        chunk = pd.read_csv(f"{CHUNKS_DIR}/chunk_1.csv")
        os.makedirs("data", exist_ok=True)
        chunk.to_csv(TRAIN_CSV, index=False)
        write_state(1)
        print(f"\nInitialised data/train.csv with chunk_1 ({len(chunk)} rows).")
        return

    # Check if all chunks have been added
    next_chunk = current + 1
    next_path = f"{CHUNKS_DIR}/chunk_{next_chunk}.csv"
    if not os.path.exists(next_path):
        print(
            f"All chunks have been added (last was chunk_{current}). No more data to stream."
        )
        return

    # Append next chunk to train.csv
    existing = pd.read_csv(TRAIN_CSV)
    new_data = pd.read_csv(next_path)
    updated = pd.concat([existing, new_data], ignore_index=True)
    updated.to_csv(TRAIN_CSV, index=False)
    write_state(next_chunk)
    print(
        f"Added chunk_{next_chunk}. data/train.csv now has {len(updated)} rows (was {len(existing)})."
    )


def emit_new_data_event():
    try:
        from prefect.events import emit_event

        emit_event(
            event="new-data-available",
            resource={"prefect.resource.id": "training-data"},
        )
        print("Prefect event 'new-data-available' emitted.")
    except Exception as exc:
        print(f"Could not emit Prefect event (is the server running?): {exc}")


if __name__ == "__main__":
    main()
    emit_new_data_event()
