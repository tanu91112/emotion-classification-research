from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from .config import config

app = FastAPI(title="Emotion Classification API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

EMOTION_LABELS = config.labels
EMOTION_EMOJIS = {"sadness": "", "joy": "", "love": "", 
                  "anger": "", "fear": "", "surprise": ""}

class TextInput(BaseModel):
    text: str

# Load model
try:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.distilbert_model, num_labels=6)
    model = PeftModel.from_pretrained(base_model, config.distilbert_path)
    tokenizer = AutoTokenizer.from_pretrained(config.distilbert_path)
    model.eval()
    loaded = True
    print(" Model loaded!")
except:
    loaded = False

@app.get("/health")
def health():
    return {"status": "healthy" if loaded else "unhealthy"}

@app.post("/predict")
def predict(input: TextInput):
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = tokenizer(input.text, padding=True, truncation=True, 
                       max_length=50, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = (time.time() - start) * 1000
    
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    pred_idx = np.argmax(probs)
    
    return {
        "text": input.text,
        "predicted_emotion": EMOTION_LABELS[pred_idx],
        "emotion_emoji": EMOTION_EMOJIS[EMOTION_LABELS[pred_idx]],
        "confidence": float(probs[pred_idx]),
        "latency_ms": round(latency, 2),
        "all_probabilities": {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


