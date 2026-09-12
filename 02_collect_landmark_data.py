"""
Day 3-4 (updated): Collect labeled hand-landmark training data.
Uses MediaPipe's Tasks API (same one that worked in 01_test_hand_tracking.py).

Requires hand_landmarker.task in the same folder as this script.

How it works:
- Set the CLASSES list below to your target signs (e.g., letters or words).
- Run the script. It shows your webcam feed with the current target label.
- Press SPACE to capture a sample of the current label (saves landmarks to CSV).
- Press 'n' to move to the next label.
- Press 'q' to quit and save.

Run with:
    python 02_collect_landmark_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# ---- CONFIG: edit this list to your target signs ----
CLASSES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
# example: replace with your full set
SAMPLES_PER_CLASS_TARGET = 300
OUTPUT_CSV = "data/landmarks.csv"
# -------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize_landmarks(hand_landmarks):
    """Convert landmarks to a flat, wrist-relative, scale-normalized vector."""
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
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    rows = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        rows = existing.values.tolist()
        print(f"Loaded {len(rows)} existing samples from {OUTPUT_CSV}")

    counts = {c: sum(1 for r in rows if r[-1] == c) for c in CLASSES}

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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        current_label = CLASSES[class_idx]
        landmark_vector = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            landmark_vector = normalize_landmarks(hand_landmarks)

        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Collected: {counts.get(current_label, 0)}/{SAMPLES_PER_CLASS_TARGET}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE=capture  n=next label  q=quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            class_idx = (class_idx + 1) % len(CLASSES)
        elif key == ord(" ") and landmark_vector is not None:
            row = list(landmark_vector) + [current_label]
            rows.append(row)
            counts[current_label] = counts.get(current_label, 0) + 1
            print(f"Captured sample for '{current_label}' "
                  f"({counts[current_label]}/{SAMPLES_PER_CLASS_TARGET})")

    cap.release()
    cv2.destroyAllWindows()

    columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} total samples to {OUTPUT_CSV}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
