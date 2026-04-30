import io
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# Global dictionary to hold the model
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Load the model from MLflow Model Registry

    print("Loading model from MLflow Registry...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", "beans-disease-classifier")
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")

    print(tracking_uri)

    model_uri = f"models:/{model_name}@{model_alias}"

    try:
        models["vit_model"] = mlflow.transformers.load_model(model_uri)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    yield
    models.clear()
    print("Model unloaded.")


app = FastAPI(lifespan=lifespan, title="ViT Image Classifier API")


class PredictionResponse(BaseModel):
    label: str
    score: float


# TODO: Implement a check that returns 200 if the model is ready, 503 otherwise.
@app.get("/health")
def health_check():
    # Check if the model object exists and is functional
    if models.get("vit_model") is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy"}, 503


# TODO Implemet /predict endpoint
# Validate that the uploaded file is an image.
# Convert bytes to a format the model accepts (PIL/Numpy).
# Return the prediction label and confidence score.
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if "vit_model" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert bytes to Image
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content)).convert("RGB")

    prediction = models["vit_model"](image)

    return {"label": prediction[0]["label"], "score": prediction[0]["score"]}
