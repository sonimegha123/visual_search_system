# build_index.py
import os
import sqlite3
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Configuration ---
IMAGE_DIR = "images"
DB_FILE = "images.db"
INDEX_FILE = "faiss_index.bin"
CLIP_MODEL = "ViT-L/14"
DEVICE = "cpu"

# --- Load CLIP model ---
print("Loading CLIP model...")
clip_model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)

# --- Load BLIP model for captions ---
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# --- Connect to DB ---
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE,
    caption TEXT,
    explanation TEXT
)
""")

conn.commit()

# --- Load images ---
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
print(f"Found {len(image_files)} images.")

all_embeddings = []

for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(IMAGE_DIR, img_file)
    try:
        # --- Load & preprocess image ---
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)

        # --- Image embedding ---
        with torch.no_grad():
            img_emb = clip_model.encode_image(image)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # --- BLIP caption generation ---
        raw_image = Image.open(img_path).convert("RGB")
        inputs = blip_processor(raw_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # --- Caption embedding with CLIP ---
        text_tokens = clip.tokenize([caption]).to(DEVICE)
        with torch.no_grad():
            caption_emb = clip_model.encode_text(text_tokens)
        caption_emb = caption_emb / caption_emb.norm(dim=-1, keepdim=True)

        # --- Combine embeddings (average) ---
        combined_emb = (img_emb + caption_emb) / 2
        combined_emb = combined_emb.cpu().numpy().astype("float32")
        all_embeddings.append(combined_emb)

        # --- Store in DB ---
        c.execute(
            "INSERT OR REPLACE INTO images (filename, caption) VALUES (?, ?)",
            (img_file, caption)
        )

    except Exception as e:
        print(f"Skipping {img_file}: {e}")

conn.commit()
conn.close()

# --- Build FAISS index ---
all_embeddings = np.vstack(all_embeddings)
embedding_dim = all_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
faiss.normalize_L2(all_embeddings)
index.add(all_embeddings)
faiss.write_index(index, INDEX_FILE)

print(f"FAISS index saved to {INDEX_FILE}")
print(f"Database saved to {DB_FILE}")
