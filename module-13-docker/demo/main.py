from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load the model
MODEL_NAME = "elastic/distilbert-base-cased-finetuned-conll03-english"
classifier = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Run NER inference
    entities = classifier(request.text)

    # Transformers returns objects that aren't always JSON-serializable
    # (like numpy floats), so we clean them up
    results = []
    for entity in entities:
        results.append(
            {
                "entity_group": entity["entity_group"],
                "score": float(entity["score"]),
                "word": entity["word"],
                "start": entity["start"],
                "end": entity["end"],
            }
        )

    return {"entities": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
