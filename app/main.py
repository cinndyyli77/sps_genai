from fastapi import APIRouter
from pydantic import BaseModel
from app.rnn_model import LSTMTextGenerator
from app.mnist_api import router as mnist_router
import spacy


app = APIRouter()

app.include_router(mnist_router)

nlp = spacy.load("en_core_web_md")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

rnn_model = LSTMTextGenerator(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = rnn_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}
