import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

nlp = spacy.load("en_core_web_sm")


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    doc = nlp(request.text)

    return {
        "entities": [
            {"word": ent.text, "entity_group": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
