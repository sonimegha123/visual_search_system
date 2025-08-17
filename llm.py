# llm.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

DEVICE = "cpu"
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def generate_explanation(query: str, caption: str, max_length: int = 100) -> str:
    prompt = (
        f"Query: '{query}'\n"
        f"Image Caption: '{caption}'\n"
        "Explain concisely why this image is relevant to the query."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
