import cv2
import mediapipe as mp
import numpy as np
import joblib

print("Starting ASL Recognition System...")

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("models/asl_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit()

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# CAMERA
# =========================
print("Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera!")
    exit()

print("Camera opened successfully!")

# =========================
# MAIN LOOP
# =========================
while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame!")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    predicted_letter = ""

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
                features = np.array(features).reshape(1, -1)

                prediction = model.predict(features)

                predicted_letter = encoder.inverse_transform(
                    prediction
                )[0]

            except Exception as e:
                print("Prediction error:", e)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # DISPLAY LETTER
    # =========================
    cv2.putText(
        frame,
        f"Letter: {predicted_letter}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "ASL Recognition",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("Program Closed.")