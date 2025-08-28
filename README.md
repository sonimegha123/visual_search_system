# 🖼️ Visual Search System  
A lightweight image search system that lets you find images using natural language queries. Built with CLIP, BLIP, FAISS, and FastAPI, and fully containerized with Docker.  
  
<img width="475" height="393" alt="Screenshot 2025-08-28 at 15 56 32" src="https://github.com/user-attachments/assets/260456f0-566a-4bed-97ff-9c55d5eb81b4" />

**How It Works**  
**Indexing (offline)**  
Generate CLIP embeddings for images.  
Generate BLIP captions for context.  
Store captions + filenames in SQLite.  
Build a FAISS index for similarity search.  
**Search (online)**  
User enters a query → encoded with CLIP text encoder.  
Search FAISS for nearest embeddings.  
Return top results with captions as explanations.  
  
**build index**  

python build_index.py  

**Tech**  
CLIP (ViT-L/14) → embeddings  
BLIP → captions  
FAISS → similarity search  
SQLite → metadata store  
FastAPI + Docker → deployment  

**Use Cases**  
Search product catalogs  
Explore photo collections  
Retrieve stock/media images  
And more...
