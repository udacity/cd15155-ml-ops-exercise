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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "beans-disease-classifier")
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")

    #TODO set tracking URI


    #TODO Get model URI using model name and alias


    #TODO Load the model using the model URI
    # Raise an exception if the model cannot be loaded


    yield
    #TODO: Clear the model from memory when the app is shutting down


#TODO Create a FastAPI app 
app = ... 


class PredictionResponse(BaseModel):
    label: str
    score: float


# TODO: Implement a check that returns 200 if the model is ready, 503 otherwise.
def health_check():
    raise NotImplementedError("Implement health check endpoint")

# TODO Implemet /predict endpoint 
# Use the response model PredictionResponse to validate the response
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
    pass
