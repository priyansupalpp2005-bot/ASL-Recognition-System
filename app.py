import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont

# Lazy imports for optional local-only features
CV2_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None

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
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        st.warning("Text-to-speech not available (pyttsx3 missing or failed).")

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
# ======================
# CAMERA + PREDICTION
# ======================

class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        # Initialize MediaPipe hands only if available (local)
        if MEDIAPIPE_AVAILABLE and CV2_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.hands = None

        self.predicted_letter = ""

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        # If MediaPipe is available (local), run detection; otherwise skip
        if self.hands is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
        else:
            results = None

        # reset predicted for this frame
        self.predicted_letter = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                features = []

                for lm in hand_landmarks.landmark:
                    features.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                try:

                    features = np.array(
                        features
                    ).reshape(1, -1)

                    prediction = model.predict(features)

                    self.predicted_letter = encoder.inverse_transform(
                        prediction
                    )[0]

                except Exception as e:
                    print("Prediction Error:", e)

                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        # Overlay prediction text. Prefer OpenCV if available, else use PIL.
        label_text = f"Letter: {self.predicted_letter if self.predicted_letter else '_'}"
        if CV2_AVAILABLE:
            cv2.putText(
                img,
                label_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out_frame = img
            out_fmt = "bgr24"
        else:
            rgb_img = img[:, :, ::-1]
            pil_img = Image.fromarray(rgb_img)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.load_default()
            draw.text((20, 50), label_text, fill=(0, 255, 0), font=font)
            out_frame = np.array(pil_img)[:, :, ::-1]
            out_fmt = "bgr24"

        return av.VideoFrame.from_ndarray(out_frame, format=out_fmt)


if run:
    ctx = webrtc_streamer(
        key="asl-camera",
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    # Safely retrieve the active transformer instance and copy latest prediction
    def _get_transformer(c):
        if not c:
            return None
        # common attribute names across versions
        for name in ("video_transformer", "video_processor", "video_transformer_instance", "video_processor_instance"):
            obj = getattr(c, name, None)
            if obj:
                return obj
        # some versions expose transformer under state
        state = getattr(c, "state", None)
        if state:
            for name in ("video_transformer", "video_processor"):
                obj = getattr(state, name, None)
                if obj:
                    return obj
        return None

    transformer = _get_transformer(ctx)
    if transformer and hasattr(transformer, "predicted_letter"):
        pred = transformer.predicted_letter
        if pred:
            st.session_state.current_letter = pred