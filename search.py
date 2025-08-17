# search.py
import os
import sqlite3
import torch
import clip
import faiss
import numpy as np
import time

from explain import explain_image
from llm import generate_explanation

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
    for idx, score in zip(indices[0], distances[0]):
        c.execute("SELECT filename, caption, explanation FROM images LIMIT 1 OFFSET ?", (int(idx),))
        row = c.fetchone()
        if row:
            filename, caption, explanation = row

            if not caption:
                image_path = os.path.join(IMAGE_DIR, filename)
                caption = explain_image(image_path)
                c.execute("UPDATE images SET caption=? WHERE filename=?", (caption, filename))
                conn.commit()

            if not explanation:
                explanation = generate_explanation(query, caption)
                c.execute("UPDATE images SET explanation=? WHERE filename=?", (explanation, filename))
                conn.commit()

            results.append({
                "filename": filename,
                "score": float(score),
                "caption": caption,
                "explanation": explanation
            })
    conn.close()

    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.2f} seconds")
    return results



