
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import torch
import re

# FastAPI application
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"

# Load saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

# HTML template location (main project folder)
templates = Jinja2Templates(directory=str(BASE_DIR))


# Input data format
class DialogueInput(BaseModel):
    dialogue: str


# Clean input text
def clean_data(text):
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip().lower()

    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# API endpoint for summarization
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(dialogue_input.dialogue)

    return {"summary": summary}


# Home page


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="text-summarizer.html",
        context={"request": request}
    )

