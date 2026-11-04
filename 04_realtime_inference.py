"""
Day 10-11 (updated): Real-time sign detection using the trained model.
Uses MediaPipe's Tasks API (same one used in the other updated scripts).

Requires hand_landmarker.task in the same folder as this script.

Run with:
    python 04_realtime_inference.py
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "models/sign_classifier.h5"
ENCODER_PATH = "models/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.30
SMOOTHING_WINDOW = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize_landmarks(hand_landmarks):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    wrist = coords[0].copy()
    coords -= wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def main():
    model = tf.keras.models.load_model(CLASSIFIER_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        display_label = ""
        confidence = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            vector = normalize_landmarks(hand_landmarks).reshape(1, -1)
            probs = model.predict(vector, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                pred_label = encoder.inverse_transform([pred_idx])[0]
                prediction_buffer.append(pred_label)
            else:
                prediction_buffer.append(None)

            valid = [p for p in prediction_buffer if p is not None]
            if valid:
                display_label = Counter(valid).most_common(1)[0][0]

            cv2.putText(frame, f"{display_label} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            prediction_buffer.append(None)

        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
