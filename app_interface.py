import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Streamlit Page Config
st.set_page_config(page_title="Kaipullai AI - Face Anti-Spoofing", layout="centered")

# Headings
st.markdown("<h2 style='text-align:center; color:white;'>🧠 Kaipullai: Face Anti-Spoofing System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Detects Real vs Spoofed Faces in Real-Time</p>", unsafe_allow_html=True)
st.markdown("---")

# Constants
MODEL_PATH = "anti_spoof_model.h5"
ENROLL_DIR = "enrolled_faces"
os.makedirs(ENROLL_DIR, exist_ok=True)

# Load Model with Caching
@st.cache_resource
def load_spoof_model():
    return load_model(MODEL_PATH)

model = load_spoof_model()

# Detection Function
def detect_spoof(frame):
    resized = cv2.resize(frame, (224, 224))  #
    normalized = resized.astype("float32") / 255.0
    face_array = img_to_array(normalized)
    face_array = np.expand_dims(face_array, axis=0)
    prediction = model.predict(face_array)[0][0]
    return "Real" if prediction > 0.8 else "Spoof"

# Enroll Function
def enroll_face(name, frame):
    filepath = os.path.join(ENROLL_DIR, f"{name}.jpg")
    cv2.imwrite(filepath, frame)
    return f"Face saved as {name}.jpg"

# Streamlit Controls
FRAME_WINDOW = st.image([])
run = st.button("Start Webcam")
enroll_name = st.text_input("Enter name to enroll")
enroll = st.button("Enroll Face")

# Webcam Streaming
if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access camera.")
            break

        result = detect_spoof(frame)

        label_color = (0, 255, 0) if result == "Real" else (0, 0, 255)
        cv2.putText(frame, f"Detection: {result}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if enroll and enroll_name:
            enroll_face(enroll_name, frame)
            st.success(f"Enrolled {enroll_name}")
            break

    cap.release()