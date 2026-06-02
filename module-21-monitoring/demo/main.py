"""
Metrics exposed at /metrics:
  - ner_requests_total            : request count (labelled by status)
  - ner_request_duration_seconds  : request latency histogram
  - error rate derived from the above in Grafana
"""

import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from transformers import pipeline

REQUEST_COUNT = Counter(
    "ner_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ner_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

MODEL_NAME = "elastic/distilbert-base-cased-finetuned-conll03-english"

print(">>> Loading model...", flush=True)
classifier = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
print(">>> Model ready.", flush=True)

app = FastAPI(title="NER API with Prometheus")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    entities = classifier(request.text)
    return {
        "entities": [
            {
                "word": e["word"],
                "entity_group": e["entity_group"],
                "score": float(e["score"]),
                "start": e["start"],
                "end": e["end"],
            }
            for e in entities
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
