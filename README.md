# Sign Language Detection Project

Real-time static sign detection using MediaPipe hand landmarks + a Keras classifier.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow (matches the day-by-day plan)

1. **`scripts/01_test_hand_tracking.py`** (Day 1-2)
   Confirms your webcam + MediaPipe pipeline works. Just draws hand landmarks live.

2. **`scripts/02_collect_landmark_data.py`** (Day 3-5)
   Edit the `CLASSES` list at the top to your target signs (e.g. `["A","B","C",...]`
   or `["hello","thanks","yes","no"]`). Run it, hold each sign, press SPACE to
   capture samples (aim for 200-500 per class), press `n` for the next class.
   Saves to `data/landmarks.csv`.

3. **`scripts/03_train_classifier.py`** (Day 6-9)
   Trains a feedforward network on the collected landmarks. Prints a
   classification report + confusion matrix. Saves the model to
   `models/sign_classifier.h5` and the label encoder to `models/label_encoder.pkl`.

4. **`scripts/04_realtime_inference.py`** (Day 10-12)
   Loads the trained model and runs live webcam predictions with a
   confidence threshold and a smoothing buffer (majority vote over recent
   frames) to reduce flicker.

## Notes

- Landmarks are normalized (wrist-relative, scale-normalized) so signs are
  recognized regardless of hand position/distance from camera — don't skip
  this step or accuracy will suffer a lot.
- This pipeline only handles **static** signs. For **dynamic/word-level**
  signs (motion matters, e.g. "hello" as a wave), you'd instead collect
  sequences of ~30 frames per sample and train an LSTM on the sequence
  instead of a single-frame Dense network — ask if you want that variant.
- If accuracy is poor for certain classes, check the confusion matrix from
  step 3, collect more/varied samples for the weak classes, and retrain.
