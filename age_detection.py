import streamlit as st
import cv2
import numpy as np
import urllib.request
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.title("🎥 Live Age Detection")

# -------- DOWNLOAD MODELS --------
def download(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

download("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", "deploy.prototxt")
download("https://github.com/opencv/opencv_3rdparty/raw/master/dnn_models/res10_300x300_ssd_iter_140000.caffemodel", "face.caffemodel")
download("https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt", "age.prototxt")
download("https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel", "age.caffemodel")

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "face.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe("age.prototxt", "age.caffemodel")
    return face_net, age_net

face_net, age_net = load_models()

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# -------- WEBRTC CONFIG --------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# -------- VIDEO PROCESSOR --------
class AgeDetector(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                     (104, 177, 123))

        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # SAFE BOUNDS FIX 🔥
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227),
                    (78.4, 87.7, 114.9),
                    swapRB=False
                )

                age_net.setInput(face_blob)
                preds = age_net.forward()

                age = AGE_LIST[preds[0].argmax()]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, age, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

        return img

# -------- START CAMERA --------
webrtc_streamer(
    key="age-detection",
    video_transformer_factory=AgeDetector,
    rtc_configuration=RTC_CONFIGURATION
)