# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from search import search
from explain import explain_image

app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "message": "Service is running"
    }
    
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
def search_api(query: str, topk: int = 5):
    results = search(query, topk)
    for r in results:
        img_path = os.path.join("images", r["filename"])
        r["explanation"] = explain_image(img_path)
    return JSONResponse(results)

