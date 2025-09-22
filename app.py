import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
import warnings

from detection_emotion import detect_emotion
from theme_classification import classify_theme
from filter_recommender import recommend_filters
from filters import apply_filter
from utils import load_image

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mood-Boosting Art Filter Recommender",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #6a3093;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(to right, #6a3093, #a044ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4a4a4a;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #6a3093, #a044ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(106, 48, 147, 0.3);
    }
    .filter-card {
        background: linear-gradient(135deg, #f5f7fa, #e4e7ec);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .filter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .color-palette {
        display: flex;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .color-box {
        width: 40px;
        height: 40px;
        margin-right: 0.5rem;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .floating {
        animation: float 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Load face detection model
def load_face_detection_model():
    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Function to detect faces in an image
def detect_faces(img, face_cascade):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

# Function to detect emotions in an image
def detect_emotion(img, faces):
    emotions = []
    
    # If no faces detected, return empty emotions
    if len(faces) == 0:
        return emotions
    
    # For each face, analyze emotion based on facial features
    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = img[y:y+h, x:x+w]
        
        # Convert to HSV color space for better emotion analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Analyze the mouth region (approximate location)
        mouth_y = int(y + h * 0.6)
        mouth_h = int(h * 0.2)
        mouth_region = img[mouth_y:mouth_y+mouth_h, x:x+w]
        
        # Analyze the eye regions (approximate locations)
        eye_y = int(y + h * 0.2)
        eye_h = int(h * 0.2)
        eye_region = img[eye_y:eye_y+eye_h, x:x+w]
        
        # Simple emotion detection based on facial features
        # Calculate average brightness of face
        avg_brightness = np.mean(face_region)
        
        # Calculate mouth openness (simplified)
        mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_edges = cv2.Canny(mouth_gray, 50, 150)
        mouth_openness = np.sum(mouth_edges) / (w * mouth_h)
        
        # Calculate eye brightness
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_brightness = np.mean(eye_gray)
        
        # Determine emotion based on features
        if mouth_openness > 5 and avg_brightness > 130:
            emotions.append("Happy")
        elif mouth_openness < 2 and avg_brightness < 100:
            emotions.append("Sad")
        elif eye_brightness > 150:
            emotions.append("Surprised")
        elif avg_brightness < 80:
            emotions.append("Angry")
        else:
            emotions.append("Neutral")
    
    return emotions

# Function to extract dominant colors
def extract_dominant_colors(img, n_colors=5):
    # Resize image for faster processing
    img = cv2.resize(img, (100, 100))
    
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)
    
    # Use K-Means to find dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the colors and their percentages
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = counts / len(pixels) * 100
    
    # Sort by percentage
    sorted_indices = np.argsort(percentages)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_percentages = percentages[sorted_indices]
    
    return sorted_colors, sorted_percentages

# Function to recommend filters based on mood and colors
def recommend_filters(emotions, colors):
    filters = []
    
    # Filter recommendations based on emotions
    if "Happy" in emotions:
        filters.extend(["Vibrant Boost", "Sunshine Glow", "Warm Summer", "Golden Hour", "Color Pop"])
    if "Sad" in emotions:
        filters.extend(["Uplift Bright", "Pastel Dreams", "Soft Blur", "Misty Morning", "Warm Film"])
    if "Surprised" in emotions:
        filters.extend(["Electric Vibes", "Bold Contrast", "Light Leak", "Cool Film", "Vibrant Boost"])
    if "Neutral" in emotions:
        filters.extend(["Classic Vintage", "Film Grain", "Sepia Tone", "Clean White", "Moody Dark"])
    if "Angry" in emotions:
        filters.extend(["Cool Tones", "Soft Blur", "Misty Morning", "Pastel Dreams", "Clean White"])
    
    # If no emotions detected (no faces), recommend based on colors
    if len(emotions) == 0:
        avg_color = np.mean(colors, axis=0)
        if np.mean(avg_color) > 180:  # Bright image
            filters.extend(["Bright & Airy", "Clean White", "Light Leak", "Sunshine Glow", "Pastel Dreams"])
        elif np.mean(avg_color) < 100:  # Dark image
            filters.extend(["Moody Dark", "Noir", "Dramatic Contrast", "Cool Tones", "Cool Film"])
        else:
            filters.extend(["Classic Vintage", "Film Grain", "Warm Film", "Cool Film", "Sepia Tone"])
    
    # Add some general filters
    filters.extend(["Classic Vintage", "Film Grain", "Warm Film", "Cool Film", "Sepia Tone"])
    
    # Remove duplicates and return
    return list(set(filters))[:12]  # Return max 12 filters

# Function to apply filter to image
def apply_filter(img, filter_name):
    # Make a copy of the image
    enhanced = img.copy().astype(np.float32) / 255.0
    
    # Apply different filters based on the filter name
    if filter_name == "Vibrant Boost":
        # Increase saturation and contrast
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.2, beta=10)
    
    elif filter_name == "Sunshine Glow":
        # Add warm yellow tint
        enhanced[:, :, 0] *= 0.9  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 1.2  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Warm Summer":
        # Add orange tint
        enhanced[:, :, 0] *= 0.8  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 1.3  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Uplift Bright":
        # Increase brightness
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.1, beta=30)
    
    elif filter_name == "Golden Hour":
        # Add golden tint
        enhanced[:, :, 0] *= 0.7  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 1.4  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Pastel Dreams":
        # Soft pastel effect
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=0.8, beta=40)
    
    elif filter_name == "Color Pop":
        # Increase saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] *= 1.5
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Electric Vibes":
        # Strong contrast and saturation
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.3, beta=5)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.7, 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif filter_name == "Bold Contrast":
        # High contrast
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.4, beta=0)
    
    elif filter_name == "Soft Blur":
        # Soft focus effect
        blurred = cv2.GaussianBlur(enhanced, (15, 15), 10)
        enhanced = cv2.addWeighted(enhanced, 0.7, blurred, 0.3, 0)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Misty Morning":
        # Add blue tint and softness
        enhanced[:, :, 0] *= 1.3  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 0.8  # Red
        blurred = cv2.GaussianBlur(enhanced, (21, 21), 10)
        enhanced = cv2.addWeighted(enhanced, 0.7, blurred, 0.3, 0)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Cool Tones":
        # Add blue tint
        enhanced[:, :, 0] *= 1.3  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 0.8  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Bright & Airy":
        # High brightness, low contrast
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=0.9, beta=50)
    
    elif filter_name == "Clean White":
        # High brightness, blue tint
        enhanced[:, :, 0] *= 1.1  # Blue
        enhanced[:, :, 1] *= 1.0  # Green
        enhanced[:, :, 2] *= 1.2  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.1, beta=30)
    
    elif filter_name == "Light Leak":
        # Simulate light leak effect
        height, width = img.shape[:2]
        leak = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.ellipse(leak, (width-100, height//2), (width//2, height), 0, 0, 360, (50, 150, 255), -1)
        enhanced = cv2.addWeighted(enhanced, 0.8, leak, 0.2, 0)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Moody Dark":
        # Darken image with blue tint
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=0.7, beta=-10)
        enhanced = enhanced.astype(np.float32) / 255.0
        enhanced[:, :, 0] *= 1.2  # Blue
        enhanced[:, :, 1] *= 0.9  # Green
        enhanced[:, :, 2] *= 0.9  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Noir":
        # Black and white with high contrast
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.3, beta=0)
    
    elif filter_name == "Dramatic Contrast":
        # Very high contrast
        enhanced = cv2.convertScaleAbs(enhanced * 255, alpha=1.5, beta=-20)
    
    elif filter_name == "Classic Vintage":
        # Sepia tone effect
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        enhanced = cv2.transform(enhanced, sepia_filter)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Film Grain":
        # Add film grain noise
        noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
        enhanced = cv2.add(enhanced, noise)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Warm Film":
        # Warm tone film effect
        enhanced[:, :, 0] *= 1.2  # Blue
        enhanced[:, :, 1] *= 1.0  # Green
        enhanced[:, :, 2] *= 0.8  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Cool Film":
        # Cool tone film effect
        enhanced[:, :, 0] *= 0.8  # Blue
        enhanced[:, :, 1] *= 1.0  # Green
        enhanced[:, :, 2] *= 1.2  # Red
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    elif filter_name == "Sepia Tone":
        # Classic sepia
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        enhanced = cv2.transform(enhanced, sepia_filter)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    else:
        # Default filter (original image)
        enhanced = cv2.convertScaleAbs(enhanced * 255)
    
    return enhanced

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Mood-Boosting Art Filter Recommender</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #6a6a6a;'>
            Upload your photo and let our AI analyze the mood and colors to recommend artistic filters 
            that will enhance your image and uplift your spirits!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Display the original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Analyze the image
            with st.spinner("Analyzing image..."):
                # Load face detection model
                face_cascade = load_face_detection_model()
                
                # Detect faces
                faces = detect_faces(img, face_cascade)
                
                # Detect emotions only if faces are found
                if len(faces) > 0:
                    emotions = detect_emotion(img, faces)
                else:
                    emotions = []
                
                # Extract dominant colors
                colors, percentages = extract_dominant_colors(img)
                
                # Recommend filters
                recommended_filters = recommend_filters(emotions, colors)
            
            # Display analysis results
            st.subheader("Image Analysis")
            
            # Display face detection results
            if len(faces) > 0:
                st.markdown(f"**Faces Detected:** {len(faces)}")
                
                # Display emotions if faces found
                if emotions:
                    st.markdown("**Detected Emotions:**")
                    emotion_html = ""
                    for emotion in emotions:
                        if emotion == "Happy":
                            color = "#4CAF50"
                        elif emotion == "Sad":
                            color = "#2196F3"
                        elif emotion == "Surprised":
                            color = "#FF9800"
                        elif emotion == "Neutral":
                            color = "#9E9E9E"
                        elif emotion == "Angry":
                            color = "#F44336"
                        else:
                            color = "#9C27B0"
                        emotion_html += f'<span class="emotion-badge" style="background-color: {color}; color: white;">{emotion}</span>'
                    st.markdown(emotion_html, unsafe_allow_html=True)
                else:
                    st.markdown("**No specific emotions detected**")
            else:
                st.markdown("**No faces detected** - Filters recommended based on color analysis")
            
            # Display color palette
            st.markdown("**Dominant Colors:**")
            color_html = '<div class="color-palette">'
            for color in colors:
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                color_html += f'<div class="color-box" style="background-color: {hex_color};" title="{hex_color}"></div>'
            color_html += '</div>'
            st.markdown(color_html, unsafe_allow_html=True)
            
            # Display recommended filters
            st.markdown("**Recommended Filters:**")
            for filter_name in recommended_filters:
                st.markdown(f'<div class="filter-card">{filter_name}</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            # Apply selected filter
            st.subheader("Apply Filter")
            
            # Create a selectbox for filter selection
            selected_filter = st.selectbox("Choose a filter to apply:", recommended_filters)
            
            # Apply the selected filter
            if st.button("Apply Filter", use_container_width=True):
                with st.spinner("Applying filter..."):
                    # Apply the filter
                    filtered_img = apply_filter(img, selected_filter)
                    
                    # Display the filtered image
                    st.subheader("Filtered Image")
                    st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Download button
                    result = Image.fromarray(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
                    buf = BytesIO()
                    result.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Filtered Image",
                        data=byte_im,
                        file_name="filtered_image.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
    
    # Storytelling section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">The Art of Emotional Well-being</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f9f7ff, #f0ebff); padding: 2rem; border-radius: 12px;'>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            Research shows that colors and visual aesthetics can significantly impact our mood and emotional well-being. 
            Our Mood-Boosting Art Filter Recommender uses advanced AI to analyze the emotional content of your photos 
            and suggests artistic filters that can enhance positive emotions or transform the mood of your images.
        </p>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            Whether you're looking to brighten a gloomy photo, add energy to a dull image, or simply explore creative ways 
            to express yourself, our tool provides personalized recommendations that combine art therapy principles with 
            cutting-edge image processing technology.
        </p>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            Experience the intersection of art, technology, and mental wellness with our innovative filter recommendation system.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()