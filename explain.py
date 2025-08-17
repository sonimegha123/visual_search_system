# explain.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

DEVICE = "cpu"

_processor, _model = None, None

def explain_image(image_path):
    global _processor, _model
    if _processor is None or _model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = _processor(raw_image, return_tensors="pt").to(DEVICE)
        out = _model.generate(**inputs)
        caption = _processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return str(e)
