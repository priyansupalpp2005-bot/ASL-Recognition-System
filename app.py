import streamlit as st
#import cv2
import mediapipe as mp
import numpy as np
import joblib
#import pyttsx3

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


def speak(text):
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ======================
# SESSION STATE
# ======================
if "word" not in st.session_state:
    st.session_state.word = ""

if "current_letter" not in st.session_state:
    st.session_state.current_letter = ""

# ======================
# UI DESIGN
# ======================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
}

.main-title{
    text-align:center;
    font-size:3rem;
    font-weight:bold;
    color:#00E5FF;
}

.sub-title{
    text-align:center;
    color:#d1d5db;
    font-size:18px;
    margin-bottom:25px;
}

.card{
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    backdrop-filter:blur(10px);
    box-shadow:0px 0px 20px rgba(0,229,255,0.25);
    margin-bottom:15px;
}

.word-box{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

[data-testid="stMetricValue"]{
    color:#00E5FF;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🤟 ASL Recognition System
</div>

<div class="sub-title">
Real-Time Sign Language Recognition Using Artificial Intelligence
</div>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("## 🤖 AI Dashboard")

    st.success("✅ Model Loaded")

    st.info("🎥 Webcam Detection")

    st.info("🧠 MediaPipe Hand Tracking")

    st.info("🔊 Text To Speech")

    st.markdown("---")

    st.markdown("""
    ### 👨‍💻 Developer
    Priyansu Pal
                
    Swati Rani Bhanja
                
    Somesh Ranjan Nayak
                
    Himani Dash
                
    """)

    

col1, col2 = st.columns(2)

with col1:
    run = st.checkbox("📷 Start Camera")

with col2:
    st.metric(
        "Current Letter",
        st.session_state.current_letter
    )

st.markdown("""
<div class="card">
<h2 style="text-align:center;color:#00E5FF;">
Generated Word
</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="card">
        <div class="word-box">
            {st.session_state.word if st.session_state.word else "_"}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("➕ Add Letter"):
        st.session_state.word += st.session_state.current_letter

with c2:
    if st.button("🔊 Speak"):
        if st.session_state.word:
            speak(st.session_state.word)

with c3:
    if st.button("🗑 Clear"):
        st.session_state.word = ""

st.markdown("""
<hr>
<center>
<p style="color:gray;">
🤟 ASL Recognition System | Powered by Python, Streamlit & MediaPipe
</p>
</center>
""", unsafe_allow_html=True)

# ======================
# CAMERA
# ======================
# FRAME_WINDOW = st.empty()

# if run:

#     mp_hands = mp.solutions.hands
#     mp_draw = mp.solutions.drawing_utils

#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     cap = cv2.VideoCapture(0)

#     while run:

#         success, frame = cap.read()

#         if not success:
#             st.error("Camera not found")
#             break

#         frame = cv2.flip(frame, 1)

#         rgb = cv2.cvtColor(
#             frame,
#             cv2.COLOR_BGR2RGB
#         )

#         results = hands.process(rgb)

#         letter = ""

#         if results.multi_hand_landmarks:

#             for hand_landmarks in results.multi_hand_landmarks:

#                 features = []

#                 for lm in hand_landmarks.landmark:
#                     features.extend([
#                         lm.x,
#                         lm.y,
#                         lm.z
#                     ])

#                 features = np.array(
#                     features
#                 ).reshape(1, -1)

#                 pred = model.predict(features)

#                 letter = encoder.inverse_transform(pred)[0]

#                 st.session_state.current_letter = letter

#                 mp_draw.draw_landmarks(
#                     frame,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS
#                 )

#         cv2.putText(
#             frame,
#             f"Letter: {letter}",
#             (20, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#         FRAME_WINDOW.image(
#             cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#             use_container_width=True
#         )

#     cap.release()
st.info("Cloud deployment version. Webcam features are available only in the local version.")