# theme_classification.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def classify_theme(image):
    """Classify the main theme of an image"""
    try:
        # Preprocess image for MobileNetV2
        resized_image = cv2.resize(image, (224, 224))
        expanded_image = np.expand_dims(resized_image, axis=0)
        processed_image = preprocess_input(expanded_image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Map predictions to broader categories
        theme_categories = {
            'portrait': ['person', 'man', 'woman', 'child', 'baby', 'face', 'bride', 'groom'],
            'landscape': ['mountain', 'beach', 'sky', 'field', 'forest', 'lake', 'ocean', 'valley', 'cliff'],
            'food': ['food', 'pizza', 'burger', 'sushi', 'cake', 'ice cream', 'pasta', 'sandwich', 'steak'],
            'animal': ['dog', 'cat', 'bird', 'fish', 'animal', 'elephant', 'lion', 'tiger', 'bear'],
            'building': ['building', 'house', 'skyscraper', 'castle', 'palace', 'home', 'church'],
            'object': ['car', 'book', 'computer', 'phone', 'furniture', 'clock', 'bottle', 'chair'],
            'nature': ['tree', 'flower', 'plant', 'garden', 'wood', 'leaf', 'petal']
        }

        # Find the best matching category
        best_score = 0
        best_category = "other"

        for _, label, confidence in decoded_predictions:
            for category, keywords in theme_categories.items():
                if any(keyword in label for keyword in keywords):
                    if confidence > best_score:
                        best_score = confidence
                        best_category = category

        return best_category
    except Exception as e:
        print(f"Theme classification error: {e}")
        return "unknown"


def get_theme_confidence(image, return_all=False):
    """Get confidence score for theme classification"""
    try:
        resized_image = cv2.resize(image, (224, 224))
        expanded_image = np.expand_dims(resized_image, axis=0)
        processed_image = preprocess_input(expanded_image)

        predictions = model.predict(processed_image, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        if return_all:
            return {label: confidence for _, label, confidence in decoded_predictions}
        else:
            return decoded_predictions[0][2]  # Confidence of top prediction
    except Exception as e:
        print(f"Confidence calculation error: {e}")
        return 0.0 if not return_all else {}
