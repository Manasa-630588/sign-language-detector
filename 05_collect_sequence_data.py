"""
Collect sequence (motion) data for J, Z, greetings, etc.

UPDATED: each frame now has 66 features:
  - 63 normalized hand-shape values (wrist-relative, scaled)
  - 3 extra values: wrist x, wrist y, hand size (a proxy for distance to camera)
These extra values let the model see WHERE the hand is and how it moves,
which separates signs like HELLO (forehead) vs THANKS (chin).

Add an "IDLE" class and record random non-sign hand movement for it.

Press SPACE = record one sample (3 sec countdown, then 30 frames)
Press n = next sign, q = quit.

Run with:
    python 05_collect_sequence_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# ---- CONFIG ----
CLASSES = ["J", "Z", "HELLO", "THANKS", "PLEASE", "YES", "NO", "SORRY", "IDLE"]
SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS_TARGET = 40
OUTPUT_DIR = "data/sequences"
# ----------------

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
    return np.concatenate([shape, extra])  # 66 values


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def count_existing_samples(label):
    label_dir = os.path.join(OUTPUT_DIR, label)
    if not os.path.exists(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith(".npy")])


def main():
    for c in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    class_idx = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    recording = False
    recorded_frames = []
    countdown_start = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        current_label = CLASSES[class_idx]
        features = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            features = extract_features(hand_landmarks)

        if countdown_start is not None:
            remaining = 3 - int(time.time() - countdown_start)
            if remaining > 0:
                cv2.putText(frame, f"Get ready: {remaining}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                countdown_start = None
                recording = True
                recorded_frames = []

        if recording:
            if features is not None:
                recorded_frames.append(features)
            cv2.putText(frame, f"RECORDING {len(recorded_frames)}/{SEQUENCE_LENGTH}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if len(recorded_frames) >= SEQUENCE_LENGTH:
                sample = np.array(recorded_frames[:SEQUENCE_LENGTH])
                existing = count_existing_samples(current_label)
                path = os.path.join(OUTPUT_DIR, current_label, f"sample_{existing:04d}.npy")
                np.save(path, sample)
                print(f"Saved sample for '{current_label}' -> {path}")
                recording = False
                recorded_frames = []

        count = count_existing_samples(current_label)
        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Collected: {count}/{SAMPLES_PER_CLASS_TARGET}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE=record  n=next label  q=quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        cv2.imshow("Sequence Data Collection", frame)
        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n") and not recording and countdown_start is None:
            class_idx = (class_idx + 1) % len(CLASSES)
        elif key == ord(" ") and not recording and countdown_start is None:
            countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Sample counts per class:")
    for c in CLASSES:
        print(f"  {c}: {count_existing_samples(c)}")


if __name__ == "__main__":
    main()
