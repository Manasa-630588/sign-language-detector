"""
Real-time detection for two-handed motion signs (132 features per frame).

UPDATED FOR 2 HANDS:
- MediaPipe configured to detect up to 2 hands (num_hands=2).
- Extracts up to 2 hands (66 values each) and zero-pads if fewer hands are visible.
- Feeds a 132-feature sequence into the sequence classifier.

Run with:
    python 07_realtime_sequence_inference.py
"""

from collections import deque
import pickle
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf

MODEL_PATH = "hand_landmarker.task"
CLASSIFIER_PATH = "models/sequence_classifier.h5"
ENCODER_PATH = "models/sequence_label_encoder.pkl"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85  # how sure the model must be
AGREE_COUNT = 3  # predictions in a row that must agree
PREDICT_EVERY_N_FRAMES = 3
HOLD_SECONDS = 1.5  # how long a detected sign stays on screen
GRACE_FRAMES = (
    45  # keep the last window this long after the hands leave
)
GAP_RESET_FRAMES = 10  # hands gone this long -> start a fresh window on return

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


def main():
  model = tf.keras.models.load_model(CLASSIFIER_PATH)
  with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

  base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
  options = vision.HandLandmarkerOptions(
      base_options=base_options,
      num_hands=2,  # UPDATED TO 2 HANDS
      running_mode=vision.RunningMode.IMAGE,
  )
  detector = vision.HandLandmarker.create_from_options(options)

  buffer = deque(maxlen=SEQUENCE_LENGTH)
  recent = deque(maxlen=AGREE_COUNT)
  display_label = ""
  display_confidence = 0.0
  hold_until = 0.0
  missing = 0
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
      if missing >= GAP_RESET_FRAMES:
        buffer.clear()
        recent.clear()
      missing = 0
      for hand_landmarks in result.hand_landmarks:
        draw_landmarks(frame, hand_landmarks)
      buffer.append(extract_two_hand_features(result))
    else:
      missing += 1
      if missing > GRACE_FRAMES:
        buffer.clear()
        recent.clear()

    frame_count += 1
    can_predict = len(buffer) == SEQUENCE_LENGTH and missing <= GRACE_FRAMES
    if can_predict and frame_count % PREDICT_EVERY_N_FRAMES == 0:
      seq = np.array(buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, -1)
      probs = model.predict(seq, verbose=0)[0]
      idx = int(np.argmax(probs))
      conf = float(probs[idx])
      label = (
          encoder.inverse_transform([idx])[0]
          if conf >= CONFIDENCE_THRESHOLD
          else None
      )
      recent.append(label)
      if len(recent) == AGREE_COUNT and len(set(recent)) == 1:
        chosen = recent[0]
        if chosen not in (None, "IDLE"):
          display_label = chosen
          display_confidence = conf
          hold_until = time.time() + HOLD_SECONDS

    if display_label and time.time() > hold_until:
      display_label = ""

    if display_label:
      cv2.putText(
          frame,
          f"{display_label} ({display_confidence:.2f})",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.2,
          (0, 255, 0),
          3,
      )

    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Motion Sign Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()