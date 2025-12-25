from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from backend.rag_pipeline import get_diagnosis
from backend.cnn.predict import predict_image

# 🔴 APP INIT (MANDATORY)
app = FastAPI(title="Healthcare RAG Diagnostic System")

# 🔴 CORS FIX (VERY IMPORTANT FOR REACT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔴 UPLOAD DIRECTORY
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "Backend is running successfully"}

# ---------------- TEXT DIAGNOSIS (RAG) ----------------
@app.post("/ask")
def ask_question(payload: dict):
    query = payload.get("query")

    if not query:
        return {"error": "Query field is required"}

    answer = get_diagnosis(query)

    return {
        "type": "Text Diagnosis",
        "answer": answer,
        "disclaimer": "AI-generated response. Not medical advice."
    }

# ---------------- IMAGE DIAGNOSIS (CNN) ----------------
@app.post("/diagnose-image")
async def diagnose_image(file: UploadFile = File(...)):
    try:
        # 1️⃣ Save uploaded image
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2️⃣ CNN prediction
        result = predict_image(file_path)

        return {
            "type": "Image Diagnosis",
            "prediction": result["prediction"],
            "confidence": f"{result['confidence']}%",
            "disclaimer": "AI-generated result. Not medical advice."
        }

    except Exception as e:
        return {"error": str(e)}
#  python -m uvicorn backend.main:app --reload