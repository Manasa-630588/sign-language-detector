"""
Collect sequence (motion) data for two-handed gestures.

UPDATED FOR 2 HANDS:
Each frame now outputs 132 features (66 per hand x 2 hands):
  - Hand 1: 63 shape values + 3 extras (wrist x, y, size)
  - Hand 2: 66 values (or padded with 0s if only 1 hand is visible)

Run with:
    python 05_collect_sequence_data.py
"""

import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_PATH = "hand_landmarker.task"

# ---- CONFIG ----
CLASSES = ["name","is","today","meet","friend","what","again","when","why","who","how","spider","cat","owl","scary","blood","help","open","sit","find","sleep","mother","father","grand mother","grand father","brother","sister","study","come","move","stop","good","morning","afternoon","night",
    "sorry",
    "ILY",
    "welcome",
    "bye",
    "10",
    "J",
    "Z",
    "HELLO",
    "THANKS",
    "PLEASE",
    "YES",
    "NO",
    "IDLE",
]
SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS_TARGET = 40
OUTPUT_DIR = "data/sequences"
# ----------------

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def extract_features_single_hand(hand_landmarks):
  coords = np.array(
      [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
  )
  wrist = coords[0].copy()
  rel = coords - wrist
  max_dist = float(np.max(np.linalg.norm(rel, axis=1)))
  shape = (rel / max_dist).flatten() if max_dist > 0 else rel.flatten()
  extra = np.array([wrist[0], wrist[1], max_dist], dtype=np.float32)
  return np.concatenate([shape, extra])  # 66 values


def extract_two_hand_features(result):
  """Extracts up to 2 hands (66 values each).

  Pads missing hands with zeros -> 132 features total.
  """
  features = []
  if result.hand_landmarks:
    detected_hands = result.hand_landmarks[:2]
    for hand in detected_hands:
      features.extend(extract_features_single_hand(hand))

  # Pad with zeros if fewer than 2 hands detected
  expected_len = 132  # 66 * 2
  while len(features) < expected_len:
    features.append(0.0)

  return np.array(features, dtype=np.float32)


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
  return len([f for f in os.listdir(label_dir) if f.endswith('.npy')])


def main():
  for c in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

  base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
  options = vision.HandLandmarkerOptions(
      base_options=base_options,
      num_hands=2,  # UPDATED TO 2 HANDS
      running_mode=vision.RunningMode.IMAGE,
  )
  detector = vision.HandLandmarker.create_from_options(options)

  class_idx = 0
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print('ERROR: Could not open webcam.')
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

    if result.hand_landmarks:
      for hand_landmarks in result.hand_landmarks:
        draw_landmarks(frame, hand_landmarks)

    features = extract_two_hand_features(result)

    if countdown_start is not None:
      remaining = 3 - int(time.time() - countdown_start)
      if remaining > 0:
        cv2.putText(
            frame,
            f'Get ready: {remaining}',
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )
      else:
        countdown_start = None
        recording = True
        recorded_frames = []

    if recording:
      recorded_frames.append(features)
      cv2.putText(
          frame,
          f'RECORDING {len(recorded_frames)}/{SEQUENCE_LENGTH}',
          (10, 100),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 0, 255),
          3,
      )

      if len(recorded_frames) >= SEQUENCE_LENGTH:
        sample = np.array(recorded_frames[:SEQUENCE_LENGTH])
        existing = count_existing_samples(current_label)
        path = os.path.join(
            OUTPUT_DIR, current_label, f'sample_{existing:04d}.npy'
        )
        np.save(path, sample)
        print(f"Saved sample for '{current_label}' -> {path}")
        recording = False
        recorded_frames = []

    count = count_existing_samples(current_label)
    cv2.putText(
        frame,
        f'Label: {current_label}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f'Collected: {count}/{SAMPLES_PER_CLASS_TARGET}',
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        'SPACE=record  n=next label  q=quit',
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    cv2.imshow('Sequence Data Collection', frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
      break
    elif key == ord('n') and not recording and countdown_start is None:
      class_idx = (class_idx + 1) % len(CLASSES)
    elif key == ord(' ') and not recording and countdown_start is None:
      countdown_start = time.time()

  cap.release()
  cv2.destroyAllWindows()
  print('\nDone. Sample counts per class:')
  for c in CLASSES:
    print(f'  {c}: {count_existing_samples(c)}')


if __name__ == '__main__':
  main()