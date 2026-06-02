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
    explainer = shap.Explainer(pipe)
    print("Ready.")
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
    shap_values = explainer([text])
    tokens = list(shap_values.data[0])
    return {
        "tokens": tokens,
        "shap_values": {
            label: [float(v) for v in shap_values.values[0, :, i]]
            for i, label in enumerate(label_names)
        },
    }


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
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return _run_shap(request.text)


# TODO: implement /predict-explain 
# run prediction and return predictions + SHAP values for the top label only
@app.post("/predict-explain")
def predict_explain(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    predictions = pipe(request.text)[0]
    top_label = max(predictions, key=lambda x: x["score"])["label"]
    shap_data = _run_shap(request.text)
    return {
        "predictions": predictions,
        "explanation": {
            "label": top_label,
            "tokens": shap_data["tokens"],
            "shap_values": shap_data["shap_values"][top_label],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
