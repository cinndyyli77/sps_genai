from fastapi import APIRouter
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

app = APIRouter()

nlp = spacy.load("en_core_web_md")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(word: str):
    doc = nlp(word)
    return {
        "word": word,
        "embedding": doc.vector.tolist()
    }