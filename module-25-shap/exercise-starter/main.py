"""
On-demand explainability API using SHAP.
"""

from contextlib import asynccontextmanager

# TODO: import shap
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_NAME = "nateraw/bert-base-uncased-emotion"

pipe = None
explainer = None
label_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, explainer, label_names
    print(f"Loading model: {MODEL_NAME}...")
    pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    label_names = [pipe.model.config.id2label[i] for i in range(len(pipe.model.config.id2label))]
    print("Initializing SHAP explainer...")
    # TODO: initialize a shap.Explainer wrapping the pipeline
    # A pipeline object can be passed directly to SHAP Explainer
    # https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
    explainer = ...

    yield
    pipe = None
    explainer = None
    label_names = None


app = FastAPI(title="Emotion Explainability API", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


# TODO: implement _run_shap
# Run the explainer on the text and return tokens and per-label SHAP values
def _run_shap(text: str) -> dict:
    raise NotImplementedError("Implement the _run_shap function to return tokens and per-label SHAP values for the input text")


@app.get("/health")
def health():
    return {"status": "healthy" if pipe else "loading", "model": MODEL_NAME}


@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return {"predictions": pipe(request.text)[0]}


# TODO: implement /explain and return token-level SHAP values for all labels
@app.post("/explain")
def explain(request: TextRequest):
    raise NotImplementedError("Implement the /explain endpoint")


# TODO: implement /predict-explain 
# run prediction and return predictions + SHAP values for the top label only
@app.post("/predict-explain")
def predict_explain(request: TextRequest):
    raise NotImplementedError("Implement the /predict-explain endpoint")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
