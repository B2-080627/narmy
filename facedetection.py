# facedetection.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("😎 Face Detection App (Pro)")

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Select Mode", ["Upload Image", "Use Camera"])

# Load Haar Cascade
@st.cache_resource
def load_model():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_model()

# -------- FUNCTION --------
def detect_faces(img):

    if img is None or img.size == 0:
        return None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img, faces

# -------- IMAGE UPLOAD --------
if mode == "Upload Image":

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img = np.array(image)

        # Convert RGB → BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        output, faces = detect_faces(img)

        if output is not None:
            st.image(output, channels="BGR", caption=f"Detected Faces: {len(faces)}")

# -------- CAMERA --------
elif mode == "Use Camera":

    picture = st.camera_input("📷 Take a picture")

    if picture is not None:

        image = Image.open(picture)
        img = np.array(image)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        output, faces = detect_faces(img)

        if output is not None:
            st.image(output, channels="BGR", caption=f"Detected Faces: {len(faces)}")

# -------- INFO --------
with st.expander("ℹ️ About"):
    st.write("""
    - Uses Haar Cascade (fast & lightweight)
    - Works on mobile 📱
    - Runs fully in browser (Streamlit)
    """)