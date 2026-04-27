import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load Model
MODEL_PATH = "models/surface_defect_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (IMPORTANT: match your training order)
CLASS_NAMES = [
    'MT_Fray',
    'MT_Free',
    'MT_Crack',
    'MT_Blowhole',
    'MT_Uneven',
    'MT_Break'
]

def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

def predict_image(img_path, model):
    img = load_img(image_path, target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]

    model = load_trained_model()
    predicted_class, confidence = predict_image(image_path,model)

    print(f"\nPrediction: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {confidence:.4f}")
