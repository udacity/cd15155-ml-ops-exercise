
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import pipeline

# TODO: Add histogram that tracks the distribution of model prediction confidence scores per request

prediction_confidence = Histogram(
    "model_prediction_confidence",
    "Distribution of model prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# TODO: Add a Prometheus counter that tracks the count of predictions per output class
prediction_class_counter = Counter(
    "model_prediction_class_total",
    "Number of predictions per output class",
    ["class_name"],
)


MODEL_NAME = "nateraw/vit-base-beans"
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print(f"Loading model: {MODEL_NAME}...")
    classifier = pipeline("image-classification", model=MODEL_NAME)
    print("Model loaded.")
    yield
    classifier = None


app = FastAPI(title="Beans Disease Classifier API", lifespan=lifespan)

# TODO: Add Prometheus instrumentation to the FastAPI app via instrumentator
# https://pypi.org/project/prometheus-fastapi-instrumentator/
# No need for implementing /metrics endpoint, it will be automatically added by the instrumentator
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "healthy" if classifier else "loading", "model": MODEL_NAME}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    predictions = classifier(image)

    #TODO extract the top prediction's confidence score and label
    top = predictions[0]
    label = top["label"]
    score = float(top["score"])

    # TODO: Observe the prediction confidence score in the histogram
    prediction_confidence.observe(score)
    # TODO: Increment the counter for the predicted class label
    prediction_class_counter.labels(class_name=label).inc()

    return {"label": label, "score": score}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
