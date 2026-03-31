from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import threading

app = FastAPI()

model = None
model_ready = False


# 📌 Request Body
class TextRequest(BaseModel):
    text: str


# 🔥 Background Model Loader (Lazy Load)
def load_model():
    global model, model_ready
    print("Loading embedding model...")
    
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    model_ready = True
    print("Model loaded ✅")


# 🚀 Startup → Background load
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_model)
    thread.start()


# 🏠 Health Check
@app.get("/")
def root():
    return {
        "status": "running",
        "model_ready": model_ready
    }


# 🔥 Embedding Endpoint
@app.post("/embedding")
def get_embedding(request: TextRequest):
    
    if not model_ready:
        return {
            "status": "loading",
            "message": "Model is loading, try again..."
        }

    embedding = model.encode(request.text).tolist()

    return {
        "text": request.text,
        "embedding": embedding,
        "dimension": len(embedding)
    }