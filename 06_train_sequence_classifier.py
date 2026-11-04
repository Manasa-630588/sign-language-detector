"""
Train an LSTM on 2-hand sequence data (132 features per frame).

UPDATED: Hand landmark shape expanded for 132 features (66 per hand).

Run with:
    python 06_train_sequence_classifier.py
"""

import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "data/sequences"
MODEL_OUT = "models/sequence_classifier.h5"
ENCODER_OUT = "models/sequence_label_encoder.pkl"

AUGMENT = True
AUG_COPIES = 10  # augmented copies made from each training sample
EPOCHS = 100
BATCH_SIZE = 16


def warp_window(seq, shift, speed):
  """Re-read a (T, F) sequence: output frame i reads original time shift + i *

  speed.
  """
  T = seq.shape[0]
  pos = np.clip(shift + np.arange(T) * speed, 0, T - 1)
  lo = np.floor(pos).astype(int)
  hi = np.minimum(lo + 1, T - 1)
  frac = (pos - lo)[:, None]
  return (seq[lo] * (1 - frac) + seq[hi] * frac).astype(np.float32)


def augment(seq, rng):
  T = seq.shape[0]
  out = warp_window(seq, rng.uniform(-12, 12), rng.uniform(0.7, 1.5))

  # Hand 1 shape noise & position
  out[:, :63] += rng.normal(0, 0.01, size=(T, 63)).astype(np.float32)
  out[:, 63:65] += rng.normal(0, 0.02, size=(1, 2)).astype(np.float32)
  out[:, 65] *= rng.uniform(0.93, 1.07)

  # Hand 2 shape noise & position (if present)
  out[:, 66:129] += rng.normal(0, 0.01, size=(T, 63)).astype(np.float32)
  out[:, 129:131] += rng.normal(0, 0.02, size=(1, 2)).astype(np.float32)
  out[:, 131] *= rng.uniform(0.93, 1.07)

  return out


def shifted_copies(X):
  """Deterministic shifted/speed-changed versions of validation samples."""
  variants = [(-10, 1.0), (10, 1.0), (0, 1.3)]
  return [np.array([warp_window(x, s, sp) for x in X]) for s, sp in variants]


def load_split():
  X_train, y_train, X_val, y_val = [], [], [], []
  for label in sorted(os.listdir(DATA_DIR)):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
      continue
    files = sorted(f for f in os.listdir(label_dir) if f.endswith('.npy'))
    if not files:
      continue
    n_val = max(1, int(round(0.2 * len(files))))
    for i, fname in enumerate(files):
      sample = np.load(os.path.join(label_dir, fname))
      if i >= len(files) - n_val:
        X_val.append(sample)
        y_val.append(label)
      else:
        X_train.append(sample)
        y_train.append(label)
  return (
      np.array(X_train, dtype=np.float32),
      np.array(y_train),
      np.array(X_val, dtype=np.float32),
      np.array(y_val),
  )


def main():
  X_train, y_train_raw, X_val, y_val_raw = load_split()
  if len(X_train) == 0:
    print('No data found. Run 05_collect_sequence_data.py first.')
    return None
  print(f'Train: {X_train.shape}  Val: {X_val.shape}')

  encoder = LabelEncoder()
  encoder.fit(np.concatenate([y_train_raw, y_val_raw]))
  y_train = encoder.transform(y_train_raw)
  y_val = encoder.transform(y_val_raw)
  n_classes = len(encoder.classes_)
  print(f'Classes: {list(encoder.classes_)}')

  rng = np.random.default_rng(42)
  if AUGMENT:
    Xs, ys = [X_train], [y_train]
    for _ in range(AUG_COPIES):
      Xs.append(np.array([augment(x, rng) for x in X_train]))
      ys.append(y_train)
    X_fit, y_fit = np.concatenate(Xs), np.concatenate(ys)
    print(f'Augmented training set: {X_fit.shape}')
  else:
    X_fit, y_fit = X_train, y_train

  shifted = shifted_copies(X_val)
  X_val_shift = np.concatenate(shifted)
  y_val_shift = np.tile(y_val, len(shifted))

  X_watch = np.concatenate([X_val, X_val_shift])
  y_watch = np.concatenate([y_val, y_val_shift])

  seq_len, n_features = X_fit.shape[1], X_fit.shape[2]
  model = models.Sequential([
      layers.Input(shape=(seq_len, n_features)),
      layers.LSTM(64, return_sequences=True),
      layers.Dropout(0.3),
      layers.LSTM(32),
      layers.Dropout(0.3),
      layers.Dense(32, activation='relu'),
      layers.Dense(n_classes, activation='softmax'),
  ])
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', patience=12, restore_best_weights=True
      ),
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5
      ),
  ]
  model.fit(
      X_fit,
      y_fit,
      validation_data=(X_watch, y_watch),
      epochs=EPOCHS,
      batch_size=BATCH_SIZE,
      callbacks=callbacks,
      verbose=2,
  )

  labels = list(range(n_classes))
  y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
  aligned_acc = float(np.mean(y_pred == y_val))
  shift_pred = np.argmax(model.predict(X_val_shift, verbose=0), axis=1)
  shifted_acc = float(np.mean(shift_pred == y_val_shift))

  print('\nClassification Report (normal validation windows):')
  print(
      classification_report(
          y_val,
          y_pred,
          labels=labels,
          target_names=encoder.classes_,
          zero_division=0,
      )
  )
  print('Confusion Matrix:')
  print(confusion_matrix(y_val, y_pred, labels=labels))
  print(f'\nAccuracy on normal validation windows:       {aligned_acc:.1%}')
  print(
      'Accuracy on time-shifted/speed-changed windows:'
      f' {shifted_acc:.1%}  (closer to live use)'
  )

  os.makedirs(os.path.dirname(MODEL_OUT) or '.', exist_ok=True)
  model.save(MODEL_OUT)
  with open(ENCODER_OUT, 'wb') as f:
    pickle.dump(encoder, f)
  print(f'\nSaved model to {MODEL_OUT}')
  print(f'Saved label encoder to {ENCODER_OUT}')
  return {'aligned': aligned_acc, 'shifted': shifted_acc}


if __name__ == '__main__':
  main()