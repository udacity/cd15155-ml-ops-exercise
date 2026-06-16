from datasets import load_dataset
import requests

ds = load_dataset("beans", split="test")

for i, item in enumerate(ds.select(range(30))):
    img = item["image"]
    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": ("temp.jpg", f, "image/jpeg")},
        )
    print(f"[{i+1}] {response.json()}")
