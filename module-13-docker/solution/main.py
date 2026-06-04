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
    model_name = os.getenv("MLFLOW_MODEL_NAME", "beans-disease-classifier")
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")

    #TODO set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    print(tracking_uri)

    #TODO Get model URI using model name and alias
    model_uri = f"models:/{model_name}@{model_alias}"

    #TODO Load the model using the model URI
    # Raise an exception if the model cannot be loaded
    try:
        models["vit_model"] = mlflow.transformers.load_model(model_uri)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    yield
    #TODO: Clear the model from memory when the app is shutting down
    models.clear()
    print("Model unloaded.")


#TODO Create a FastAPI app
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
    raise HTTPException(status_code=503, detail={"status": "unhealthy", "model_loaded": False})


# TODO Implemet /predict endpoint
# Use the response model PredictionResponse to validate the response
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    #TODO Check if the model is loaded and return 503 if not
    if "vit_model" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    #TODO Read the uploaded image
    #Hint: https://fastapi.tiangolo.com/tutorial/request-files/
    request_object_content = await file.read()

    #TODO Convert bytes to PIL Image and ensure it's in RGB format
    image = Image.open(io.BytesIO(request_object_content)).convert("RGB")

    #TODO Run inference
    prediction = models["vit_model"](image)

    #TODO Return the predicted label and confidence score
    return {"label": prediction[0]["label"], "score": prediction[0]["score"]}
