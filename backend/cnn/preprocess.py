import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # 🔥 GRAYSCALE → RGB FIX
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img
