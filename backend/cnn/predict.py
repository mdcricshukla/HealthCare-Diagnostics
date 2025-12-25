import numpy as np
from tensorflow.keras.models import load_model
from .preprocess import preprocess_image
import os

MODEL_PATH = os.path.join("backend", "cnn", "model.h5")

# Load model once
model = load_model(MODEL_PATH)

# ⚠️ Class order MUST match training folders
CLASSES = ["COVID", "NORMAL", "Viral_Pneumonia"]

def predict_image(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)[0]

    idx = np.argmax(preds)

    return {
        "prediction": CLASSES[idx],
        "confidence": round(float(preds[idx]) * 100, 2)
    }
