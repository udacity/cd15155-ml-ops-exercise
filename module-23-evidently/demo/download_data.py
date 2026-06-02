"""
Download NYC Airbnb listings

Run:
    python download_data.py
"""

import gzip
import os
import shutil
import urllib.request

URLS = {
    "listings_2025_march.csv": "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-05-01/data/listings.csv.gz",
    "listings_2025_november.csv": "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-11-01/data/listings.csv.gz",
}

os.makedirs("data", exist_ok=True)

for filename, url in URLS.items():
    dest = os.path.join("data", filename)
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        continue

    gz_path = dest + ".gz"
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(gz_path, "wb") as f:
        shutil.copyfileobj(response, f)

    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

    size_mb = os.path.getsize(dest) / 1_000_000
    print(f"Saved {dest} ({size_mb:.1f} MB)")

print("Done.")
