# emotion_detection.py
import cv2
import numpy as np
from deepface import DeepFace

def detect_faces(image):
    """Detect faces in an image"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

def analyze_emotions(image, faces):
    """Analyze emotions for each detected face"""
    emotions = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Analyze emotion using DeepFace
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            dominant_emotion = analysis[0]['dominant_emotion']
            emotions.append(dominant_emotion)
        except Exception as e:
            # If emotion detection fails, skip this face
            print(f"Emotion analysis error: {e}")
            continue
    
    return emotions

def get_overall_emotion(emotions):
    """Determine overall emotion from multiple faces"""
    if not emotions:
        return "neutral"
    
    # Count occurrences of each emotion
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Return the most common emotion
    return max(emotion_counts, key=emotion_counts.get)

def draw_emotion_analysis(image, faces, emotions):
    """Draw emotion analysis on the image"""
    for i, ((x, y, w, h), emotion) in enumerate(zip(faces, emotions)):
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 155, 255), 2)
        
        # Add emotion label
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 155, 255), 2)
    
    return image
def detect_emotion(image):
    """Wrapper function to detect overall emotion from an image"""
    faces = detect_faces(image)
    emotions = analyze_emotions(image, faces)
    return get_overall_emotion(emotions)


