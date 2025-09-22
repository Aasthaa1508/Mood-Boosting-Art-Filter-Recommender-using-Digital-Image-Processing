# utils.py
import cv2
import numpy as np
from PIL import Image
import io
import time

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return image objects"""
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    image_np = np.array(image)
    
    # Convert from RGBA to RGB if needed
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    return image, image_np

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    else:
        return f"{seconds:.2f} seconds"

def analyze_multiple_images(uploaded_files):
    """Analyze multiple images and return results"""
    results = []
    
    for uploaded_file in uploaded_files:
        # Process each image
        image, image_np = save_uploaded_file(uploaded_file)
        processed_image = preprocess_image(image_np)
        
        # Perform analysis
        dominant_colors = extract_dominant_colors(processed_image)
        color_mood = analyze_image_mood(dominant_colors)
        
        faces = detect_faces(image_np)
        emotions = analyze_emotions(image_np, faces)
        overall_emotion = get_overall_emotion(emotions)
        
        theme = classify_theme(image_np)
        mood_score = get_mood_score(color_mood, overall_emotion)
        
        results.append({
            'image': image_np,
            'color_mood': color_mood,
            'dominant_emotion': overall_emotion,
            'theme': theme,
            'mood_score': mood_score
        })
    
    return results

def load_image(file):
    """Load an image from file bytes into OpenCV format"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

