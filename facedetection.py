# live_face_rgb.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(layout="wide")
st.title("🎥 Live Face Detection + RGB Effects")

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Settings")

effect = st.sidebar.selectbox(
    "Select RGB Effect",
    ["Normal", "Red", "Green", "Blue", "Random RGB"]
)

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_model()

# -------- FUNCTION --------
def apply_effect(frame, effect):

    if effect == "Red":
        frame[:, :, 1] = 0
        frame[:, :, 2] = 0

    elif effect == "Green":
        frame[:, :, 0] = 0
        frame[:, :, 2] = 0

    elif effect == "Blue":
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0

    elif effect == "Random RGB":
        frame = np.random.randint(0, 255, frame.shape, dtype=np.uint8)

    return frame

def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, faces

# -------- LIVE CAMERA --------
run = st.checkbox("▶️ Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()

    if not ret:
        st.error("Camera not working")
        break

    # Apply RGB effect
    frame = apply_effect(frame, effect)

    # Detect faces
    frame, faces = detect_faces(frame)

    # Convert BGR → RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

    time.sleep(0.03)

cap.release()