import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# app
app = FastAPI(title="Text Summarizer App", description="Text Summarization with T5 Hugging Face", version="1.0")

# device
if torch.cuda.is_available() :
    device = torch.device("cuda")
else :
    device = torch.device("cpu")

# model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model").to(device)
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# set templates
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Base Model
class DialogueInput(BaseModel) :
    dialogue : str

# clean data
def clean_data(text:str) -> str :
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return(text)

# summarize
def summarize_dialogue(dialogue:str) -> str :
    text = clean_data(dialogue)
    inputs = tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    target_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(target_ids[0], skip_special_tokens=True)
    return(summary)

# setting endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request : Request) :
    return(
        templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request" : request}
        )
    )

@app.post("/summarize")
async def summarize(dialogue : DialogueInput) :
    dialogue = dialogue.dialogue
    summary = summarize_dialogue(dialogue)
    return({"summary" : summary})

