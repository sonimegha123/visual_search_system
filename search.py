# search.py
import os
import sqlite3
import torch
import clip
import faiss
import time



# --- Config ---
IMAGE_DIR = "images"
DB_FILE = "images.db"
INDEX_FILE = "faiss_index.bin"
MODEL_NAME = "ViT-L/14"
DEVICE = "cpu"

# --- Load CLIP & FAISS once ---
print("Loading CLIP model...")
clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
index = faiss.read_index(INDEX_FILE)
print(f"Loaded FAISS index with {index.ntotal} vectors of dimension {index.d}")

# --- Search function ---
def search(query, top_k=5):
    start_time = time.time()
    text_tokens = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy().astype("float32")

    distances, indices = index.search(text_embedding, top_k)

    # Fetch filenames + captions from DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    results = []
    for idx, score in zip(indices[0], distances[0]):
        c.execute("SELECT filename, caption FROM images LIMIT 1 OFFSET ?", (int(idx),))
        row = c.fetchone()
        if row:
            results.append({"filename": row[0], "score": float(score), "caption": row[1]})
    conn.close()
    end_time = time.time()  # end timer
    print(f"Search completed in {end_time - start_time:.3f} seconds")  # print latency

    return results
