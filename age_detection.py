import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧠 Age Detection App")

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    face_model = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    age_model = cv2.dnn.readNetFromCaffe(
        "age_deploy.prototxt",
        "age_net.caffemodel"
    )

    return face_model, age_model

face_net, age_net = load_models()

# Age groups
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# -------- DETECTION FUNCTION --------
def detect_age(image):

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            results.append((x1, y1, x2, y2, age))

    return img, results

# -------- UI --------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    img, results = detect_age(image)

    for (x1, y1, x2, y2, age) in results:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, age, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(img, channels="BGR", caption="Detected Age")