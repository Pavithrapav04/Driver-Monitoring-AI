import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
import pygame
import os
import csv


# Page Setup
st.set_page_config(page_title="Driver AI", layout="wide")
st.title("🚗 Driver Monitoring AI System")
st.caption("Real-time safety monitoring")


# Session State
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False


# Sidebar Controls
if st.sidebar.button("▶ Start Camera"):
    st.session_state.camera_on = True

if st.sidebar.button("⏹ Stop Camera"):
    st.session_state.camera_on = False


# Audio System
pygame.mixer.init()
current_sound = None

def play_sound(file):
    global current_sound
    if not os.path.exists(file):
        return
    if current_sound != file:
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        current_sound = file

def stop_sound():
    global current_sound
    pygame.mixer.music.stop()
    current_sound = None


# Logging System
log_file = "driver_log.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Time", "Event", "Risk"])

def log_event(event, risk):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%H:%M:%S"), event, risk])


# Load Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# MediaPipe Setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)


# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308


# Feature Functions
def calculate_ear(landmarks, eye_idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts

    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h1 = np.linalg.norm(np.array(p1) - np.array(p4))

    return (v1 + v2) / (2.0 * h1)


def calculate_mar(landmarks, w, h):
    upper = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
    lower = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
    left = np.array([landmarks[LEFT_MOUTH].x * w, landmarks[LEFT_MOUTH].y * h])
    right = np.array([landmarks[RIGHT_MOUTH].x * w, landmarks[RIGHT_MOUTH].y * h])

    return np.linalg.norm(upper - lower) / np.linalg.norm(left - right)


# Settings
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
DROWSY_TIME = 1.2
LOOK_THRESHOLD = 0.035


# Layout
col1, col2 = st.columns([2, 1])
frame_box = col1.empty()
status_box = col2.empty()


# Camera Helper
def get_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    return None


# Main App
if st.session_state.camera_on:

    cap = get_camera()

    if cap is None:
        st.error("❌ Camera not found")
        st.stop()

    blink_start = None
    look_start = None
    last_alert_time = 0
    cooldown = 3

    while st.session_state.camera_on:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera error")
            break

        h, w, _ = frame.shape

        phone = False
        drowsy = False
        yawn = False
        distracted = False

        # YOLO Detection
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if name == "cell phone":
                    phone = True

        # Face Processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:

                ear = (calculate_ear(face.landmark, LEFT_EYE, w, h) +
                       calculate_ear(face.landmark, RIGHT_EYE, w, h)) / 2

                mar = calculate_mar(face.landmark, w, h)

                if ear < EAR_THRESHOLD:
                    if blink_start is None:
                        blink_start = time.time()
                    elif time.time() - blink_start > DROWSY_TIME:
                        drowsy = True
                else:
                    blink_start = None

                if mar > MAR_THRESHOLD:
                    yawn = True

                left_eye_x = np.mean([face.landmark[i].x for i in LEFT_EYE])
                right_eye_x = np.mean([face.landmark[i].x for i in RIGHT_EYE])

                eye_center_x = (left_eye_x + right_eye_x) / 2
                face_center_x = face.landmark[1].x

                diff = abs(eye_center_x - face_center_x)

                if diff > LOOK_THRESHOLD:
                    if look_start is None:
                        look_start = time.time()
                    elif time.time() - look_start > 1.2:
                        distracted = True
                else:
                    look_start = None

                cv2.putText(frame, f"LookDiff: {diff:.3f}", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        risk = 0
        if drowsy: risk += 3
        if phone: risk += 2
        if distracted: risk += 2
        if yawn: risk += 1

        if risk >= 5:
            level = "HIGH 🔴"
        elif risk >= 3:
            level = "MEDIUM 🟡"
        else:
            level = "LOW 🟢"

        current_time = time.time()

        if current_time - last_alert_time > cooldown:
            if drowsy:
                play_sound("drowsy.wav")
                log_event("Drowsiness", risk)
                last_alert_time = current_time

            elif phone:
                play_sound("phone.wav")
                log_event("Phone Usage", risk)
                last_alert_time = current_time

            elif distracted:
                play_sound("attention.wav")
                log_event("Distracted", risk)
                last_alert_time = current_time

            elif yawn:
                play_sound("attention.wav")
                log_event("Yawning", risk)
                last_alert_time = current_time

            else:
                stop_sound()

        frame_box.image(frame, channels="BGR")

        status_box.markdown(f"""
### 📊 Status

😴 Drowsy: `{drowsy}`  
📱 Phone: `{phone}`  
😵 Distracted: `{distracted}`  
😮 Yawning: `{yawn}`  

---

### ⚠️ Risk Level  
**{level}**  
Score: `{risk}`
""")

        time.sleep(0.03)

    cap.release()

else:
    st.info("👉 Click 'Start Camera' to begin monitoring")