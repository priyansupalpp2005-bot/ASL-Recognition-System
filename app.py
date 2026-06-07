import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("models/asl_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ======================
# TTS
# ======================
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ======================
# SESSION STATE
# ======================
if "word" not in st.session_state:
    st.session_state.word = ""

if "current_letter" not in st.session_state:
    st.session_state.current_letter = ""

# ======================
# UI
# ======================
st.title("🤟 ASL Recognition System")

st.write("Show a hand sign to the webcam and build words.")

col1, col2 = st.columns(2)

with col1:
    run = st.checkbox("Start Camera")

with col2:
    st.metric("Current Letter", st.session_state.current_letter)

st.subheader("Current Word")
st.write(f"### {st.session_state.word}")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Add Letter"):
        st.session_state.word += st.session_state.current_letter

with c2:
    if st.button("Speak"):
        if st.session_state.word:
            speak(st.session_state.word)

with c3:
    if st.button("Clear"):
        st.session_state.word = ""

# ======================
# CAMERA
# ======================
FRAME_WINDOW = st.image([])

if run:

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    while run:

        success, frame = cap.read()

        if not success:
            st.error("Camera not found")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        letter = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                features = []

                for lm in hand_landmarks.landmark:
                    features.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                features = np.array(
                    features
                ).reshape(1, -1)

                pred = model.predict(features)

                letter = encoder.inverse_transform(pred)[0]

                st.session_state.current_letter = letter

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            frame,
            f"Letter: {letter}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    cap.release()