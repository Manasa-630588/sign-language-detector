"""
Streamlit web app: real-time sign language detection (runs on localhost).

Run from the project folder with:
    streamlit run app.py

Needs, in the same folder: hand_landmarker.task, models/sign_classifier.h5,
models/label_encoder.pkl and (for Motion mode) models/sequence_classifier.h5
and models/sequence_label_encoder.pkl.
"""

import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
import streamlit as st
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from engine import (
    LetterAccumulator,
    MotionPredictor,
    StaticPredictor,
    append_token,
    draw_landmarks,
)

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")


class FastModel:
    """Wraps a Keras model in tf.function so repeated single-sample calls are faster."""

    def __init__(self, model):
        self._fn = tf.function(
            lambda x: model(x, training=False), reduce_retracing=True
        )

    def __call__(self, x, training=False):
        return self._fn(tf.convert_to_tensor(x))


@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    models_dir = BASE / "models"
    static_model = FastModel(tf.keras.models.load_model(
        str(models_dir / "sign_classifier.h5"), compile=False
    ))
    with open(models_dir / "label_encoder.pkl", "rb") as f:
        static_enc = pickle.load(f)

    motion_model = motion_enc = None
    mm = models_dir / "sequence_classifier.h5"
    me = models_dir / "sequence_label_encoder.pkl"
    if mm.exists() and me.exists():
        motion_model = FastModel(tf.keras.models.load_model(str(mm), compile=False))
        with open(me, "rb") as f:
            motion_enc = pickle.load(f)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(BASE / "hand_landmarker.task")
        ),
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return {
        "static_model": static_model,
        "static_enc": static_enc,
        "motion_model": motion_model,
        "motion_enc": motion_enc,
        "detector": detector,
    }


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ---------- state ----------
res = load_resources()
ss = st.session_state
if "sentence" not in ss:
    ss.sentence = ""
if "cap" not in ss:
    ss.cap = None
if "engines" not in ss:
    ss.engines = {
        "static": StaticPredictor(res["static_model"], res["static_enc"]),
        "acc": LetterAccumulator(),
        "motion": (
            MotionPredictor(res["motion_model"], res["motion_enc"])
            if res["motion_model"] is not None
            else None
        ),
    }
eng = ss.engines


def reset_engines():
    eng["static"].reset()
    eng["acc"].reset()
    if eng["motion"] is not None:
        eng["motion"].reset()


def add_space():
    ss.sentence += " "


def backspace():
    ss.sentence = ss.sentence[:-1]


def clear_sentence():
    ss.sentence = ""


# ---------- sidebar ----------
st.sidebar.title("Controls")
st.sidebar.radio("Mode", ["Alphabet", "Motion"], key="mode", on_change=reset_engines)
st.sidebar.toggle("Camera on", key="camera_on")
st.sidebar.slider("Alphabet confidence", 0.50, 0.99, 0.75, 0.01, key="static_thr")
st.sidebar.slider("Motion confidence", 0.50, 0.99, 0.85, 0.01, key="motion_thr")
st.sidebar.slider("Hold time to add a letter (seconds)", 0.5, 3.0, 1.0, 0.1, key="hold_time")
st.sidebar.caption(
    "Alphabet: hold a letter steady until the bar fills. "
    "Motion: do the sign, then keep your hand in view for a moment."
)

# ---------- layout ----------
st.title("Real-Time Sign Language Detector")
left, right = st.columns([3, 2])
with left:
    frame_slot = st.empty()
with right:
    st.subheader("Detected sign")
    pred_slot = st.empty()
    conf_slot = st.empty()
    hold_slot = st.empty()
    st.subheader("Sentence")
    sentence_slot = st.empty()
    b1, b2, b3 = st.columns(3)
    b1.button("Space", on_click=add_space)
    b2.button("Backspace", on_click=backspace)
    b3.button("Clear", on_click=clear_sentence)

with st.expander("Supported signs"):
    st.write("**Alphabet:** " + ", ".join(str(c) for c in res["static_enc"].classes_))
    if res["motion_enc"] is not None:
        words = [str(c) for c in res["motion_enc"].classes_ if c != "IDLE"]
        st.write("**Motion:** " + ", ".join(words))
    else:
        st.write("Motion model not found (train it with 06_train_sequence_classifier.py).")


def show_sentence():
    if ss.sentence:
        sentence_slot.markdown(f"## {ss.sentence}▌")
    else:
        sentence_slot.markdown("## _(start signing)_")


show_sentence()

if ss.mode == "Motion" and eng["motion"] is None:
    frame_slot.warning("Motion model not found. Train it first, or use Alphabet mode.")
    st.stop()

# ---------- main loop ----------
if ss.camera_on:
    if ss.cap is None:
        ss.cap = open_camera()
    cap = ss.cap
    if cap is None:
        frame_slot.error(
            "Could not open the webcam. Close other programs using it "
            "(like the other scripts), then toggle the camera off and on."
        )
        st.stop()

    detector = res["detector"]
    frame_idx = 0
    fps = 0.0
    last_time = time.time()
    while ss.camera_on:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hand = result.hand_landmarks[0] if result.hand_landmarks else None
        if hand is not None:
            draw_landmarks(frame, hand)

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        eng["static"].threshold = ss.static_thr
        eng["acc"].hold_seconds = ss.hold_time
        if eng["motion"] is not None:
            eng["motion"].threshold = ss.motion_thr

        if ss.mode == "Alphabet":
            label, conf = eng["static"].update(hand)
            progress, added = eng["acc"].update(label, now)
            if added:
                ss.sentence = append_token(ss.sentence, added)
                show_sentence()
            hold_text = "Hold steady to add the letter"
        else:
            label, conf, event = eng["motion"].update(hand, now)
            progress = eng["motion"].fill
            if event:
                ss.sentence = append_token(ss.sentence, event)
                show_sentence()
            hold_text = "Reading motion..."

        # send a smaller JPEG to the browser (much cheaper than a full-size PNG)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, int(h * 480 / w)))
        cv2.putText(small, f"FPS: {fps:.0f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        frame_slot.image(small, channels="BGR", output_format="JPEG")
        if frame_idx % 4 == 0:
            pred_slot.markdown(f"# {label}" if label else "# —")
            conf_slot.progress(min(max(conf, 0.0), 1.0), text=f"Confidence {conf:.0%}")
            hold_slot.progress(min(max(progress, 0.0), 1.0), text=hold_text)
        frame_idx += 1
else:
    if ss.cap is not None:
        ss.cap.release()
        ss.cap = None
    frame_slot.info("Camera is off. Switch it on in the sidebar to start.")
