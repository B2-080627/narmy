import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(layout="wide")
st.title("🎥 Live Face Detection + RGB Effects")

# -------- SIDEBAR --------
st.sidebar.title("⚙️ Settings")

effect = st.sidebar.selectbox(
    "Select RGB Effect",
    ["Normal", "Red", "Green", "Blue", "Random RGB"]
)

# -------- WEBRTC CONFIG --------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# -------- FACE DETECTOR CLASS --------
class FaceRGB(VideoTransformerBase):

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def apply_effect(self, frame):

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

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Apply RGB effect
        img = self.apply_effect(img)

        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

# -------- START STREAM --------
webrtc_streamer(
    key="face-rgb",
    video_transformer_factory=FaceRGB,
    rtc_configuration=RTC_CONFIGURATION
)