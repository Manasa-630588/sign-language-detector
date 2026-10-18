"""
Real-time detection for motion signs.

UPDATED:
- uses the same 66 features per frame as 05_collect_sequence_data.py
- only predicts when a hand was visible for most of the last 30 frames
- shows a sign only after 3 predictions in a row agree (less flicker)
- the IDLE class means "no sign", so nothing is displayed

Run with:
    python 07_realtime_sequence_inference.py
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "models/sequence_classifier.h5"
ENCODER_PATH = "models/sequence_label_encoder.pkl"
SEQUENCE_LENGTH = 30
NUM_FEATURES = 66
CONFIDENCE_THRESHOLD = 0.85
PREDICT_EVERY_N_FRAMES = 3
AGREE_COUNT = 3
MIN_HAND_FRACTION = 0.6

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def extract_features(hand_landmarks):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    wrist = coords[0].copy()
    rel = coords - wrist
    max_dist = float(np.max(np.linalg.norm(rel, axis=1)))
    shape = (rel / max_dist).flatten() if max_dist > 0 else rel.flatten()
    extra = np.array([wrist[0], wrist[1], max_dist], dtype=np.float32)
    return np.concatenate([shape, extra])


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


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

    buffer = deque(maxlen=SEQUENCE_LENGTH)
    present = deque(maxlen=SEQUENCE_LENGTH)
    recent = deque(maxlen=AGREE_COUNT)
    display_label = ""
    display_confidence = 0.0
    frame_count = 0

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

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            buffer.append(extract_features(hand_landmarks))
            present.append(1)
        else:
            buffer.append(np.zeros(NUM_FEATURES, dtype=np.float32))
            present.append(0)

        frame_count += 1
        if len(buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N_FRAMES == 0:
            if sum(present) < MIN_HAND_FRACTION * SEQUENCE_LENGTH:
                display_label = ""
                recent.clear()
            else:
                seq = np.array(buffer).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                probs = model.predict(seq, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                label = encoder.inverse_transform([idx])[0] if conf >= CONFIDENCE_THRESHOLD else None
                recent.append(label)
                if len(recent) == AGREE_COUNT and len(set(recent)) == 1:
                    chosen = recent[0]
                    display_label = "" if chosen in (None, "IDLE") else chosen
                    display_confidence = conf

        if display_label:
            cv2.putText(frame, f"{display_label} ({display_confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Motion Sign Detection", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
