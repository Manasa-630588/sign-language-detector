"""
Detection logic for the Streamlit app (no camera or UI code in here).

- StaticPredictor:    alphabet letters from a single frame (63 features)
- LetterAccumulator:  adds a letter to the sentence once it is held steady
- MotionPredictor:    motion signs from a 30-frame window (66 features)
- append_token:       adds a letter or word to the sentence text
"""

from collections import Counter, deque

import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def static_features(hand_landmarks):
    """63 values: wrist-relative, scaled hand shape (same as the alphabet training)."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    coords -= coords[0].copy()
    max_dist = float(np.max(np.linalg.norm(coords, axis=1)))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def motion_features(hand_landmarks):
    """66 values: hand shape + wrist x, wrist y, hand size (same as motion training)."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    wrist = coords[0].copy()
    rel = coords - wrist
    max_dist = float(np.max(np.linalg.norm(rel, axis=1)))
    shape = (rel / max_dist).flatten() if max_dist > 0 else rel.flatten()
    extra = np.array([wrist[0], wrist[1], max_dist], dtype=np.float32)
    return np.concatenate([shape, extra])


def draw_landmarks(frame_bgr, hand_landmarks):
    import cv2  # imported here so this file can be tested without OpenCV

    h, w, _ = frame_bgr.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[s], points[e], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame_bgr, p, 4, (0, 0, 255), -1)


def append_token(sentence, token):
    """Single letters are appended directly, words get spaces around them."""
    if len(token) == 1:
        return sentence + token
    if sentence and not sentence.endswith(" "):
        sentence += " "
    return sentence + token + " "


class StaticPredictor:
    def __init__(self, model, encoder, threshold=0.75, window=8, min_votes=3):
        self.model = model
        self.encoder = encoder
        self.threshold = threshold
        self.min_votes = min_votes
        self.recent = deque(maxlen=window)

    def reset(self):
        self.recent.clear()

    def update(self, hand):
        """Returns (stable_label_or_empty_string, confidence)."""
        if hand is None:
            self.recent.clear()
            return "", 0.0
        x = static_features(hand)[None, :]
        probs = np.asarray(self.model(x, training=False))[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf >= self.threshold:
            self.recent.append(str(self.encoder.inverse_transform([idx])[0]))
        else:
            self.recent.append(None)
        valid = [p for p in self.recent if p is not None]
        if not valid:
            return "", conf
        label, count = Counter(valid).most_common(1)[0]
        if count < self.min_votes:
            return "", conf
        return label, conf


class LetterAccumulator:
    """Adds a letter after it has been held for hold_seconds. To type the same
    letter twice, drop your hand (or change sign) for release_seconds."""

    def __init__(self, hold_seconds=1.0, release_seconds=0.5):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = 0.0
        self.locked = False
        self.empty_since = None

    def _progress(self, now):
        if self.candidate is None:
            return 0.0
        if self.locked:
            return 1.0
        return min(1.0, (now - self.since) / self.hold_seconds)

    def update(self, label, now):
        """Returns (progress 0..1, letter_to_add_or_None)."""
        if not label:
            if self.empty_since is None:
                self.empty_since = now
            if now - self.empty_since >= self.release_seconds:
                self.candidate = None
                self.locked = False
                return 0.0, None
            return self._progress(now), None
        self.empty_since = None
        if label != self.candidate:
            self.candidate = label
            self.since = now
            self.locked = False
            return 0.0, None
        if self.locked:
            return 1.0, None
        progress = self._progress(now)
        if progress >= 1.0:
            self.locked = True
            return 1.0, label
        return progress, None


class MotionPredictor:
    """Same logic as 07_realtime_sequence_inference.py, plus an 'event' that
    fires once per detected sign so the app can add it to the sentence."""

    def __init__(self, model, encoder, seq_len=30, threshold=0.85, agree=3,
                 predict_every=3, hold_seconds=1.5, grace_frames=45,
                 gap_reset_frames=10, cooldown_seconds=2.0):
        self.model = model
        self.encoder = encoder
        self.seq_len = seq_len
        self.threshold = threshold
        self.agree = agree
        self.predict_every = predict_every
        self.hold_seconds = hold_seconds
        self.grace = grace_frames
        self.gap_reset = gap_reset_frames
        self.cooldown = cooldown_seconds
        self.buffer = deque(maxlen=seq_len)
        self.recent = deque(maxlen=agree)
        self.reset()

    def reset(self):
        self.buffer.clear()
        self.recent.clear()
        self.missing = 0
        self.frame_count = 0
        self.display_label = ""
        self.display_conf = 0.0
        self.hold_until = 0.0
        self.last_event_label = None
        self.last_event_time = -1e9
        self.locked_label = None

    @property
    def fill(self):
        return len(self.buffer) / self.seq_len

    def _clear_window(self):
        self.buffer.clear()
        self.recent.clear()
        self.locked_label = None

    def update(self, hand, now):
        """Returns (display_label, confidence, event_label_or_None)."""
        if hand is not None:
            if self.missing >= self.gap_reset:
                self._clear_window()
            self.missing = 0
            self.buffer.append(motion_features(hand))
        else:
            self.missing += 1
            if self.missing > self.grace:
                self._clear_window()

        self.frame_count += 1
        event = None
        can_predict = (len(self.buffer) == self.seq_len
                       and self.missing <= self.grace
                       and self.frame_count % self.predict_every == 0)
        if can_predict:
            seq = np.asarray(self.buffer, dtype=np.float32)[None, :, :]
            probs = np.asarray(self.model(seq, training=False))[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = (str(self.encoder.inverse_transform([idx])[0])
                     if conf >= self.threshold else None)
            self.recent.append(label)
            if label != self.locked_label:
                self.locked_label = None
            if len(self.recent) == self.agree and len(set(self.recent)) == 1:
                chosen = self.recent[0]
                if chosen not in (None, "IDLE"):
                    self.display_label = chosen
                    self.display_conf = conf
                    self.hold_until = now + self.hold_seconds
                    is_new = chosen != self.last_event_label
                    cooled = now - self.last_event_time >= self.cooldown
                    if self.locked_label is None and (is_new or cooled):
                        event = chosen
                        self.last_event_label = chosen
                        self.last_event_time = now
                        self.locked_label = chosen

        if self.display_label and now > self.hold_until:
            self.display_label = ""
        return self.display_label, self.display_conf, event
