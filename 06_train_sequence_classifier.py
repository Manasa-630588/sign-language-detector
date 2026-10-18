"""
Train an LSTM on sequence data (66 features per frame).

UPDATED: the validation set is the LAST 20% of each sign's recordings
(chronological), not a random split. Back-to-back samples are near
duplicates, so a random split gives inflated scores. This one is more honest.

Run with:
    python 06_train_sequence_classifier.py
"""

import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "data/sequences"
MODEL_OUT = "models/sequence_classifier.h5"
ENCODER_OUT = "models/sequence_label_encoder.pkl"


def load_split():
    X_train, y_train, X_val, y_val = [], [], [], []
    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
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
    return (np.array(X_train, dtype=np.float32), np.array(y_train),
            np.array(X_val, dtype=np.float32), np.array(y_val))


def main():
    X_train, y_train_raw, X_val, y_val_raw = load_split()
    if len(X_train) == 0:
        print("No data found. Run 05_collect_sequence_data.py first.")
        return
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_raw, y_val_raw]))
    y_train = encoder.transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    print(f"Classes: {list(encoder.classes_)}")

    seq_len, num_features = X_train.shape[1], X_train.shape[2]

    model = models.Sequential([
        layers.Input(shape=(seq_len, num_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-5),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=8, callbacks=callbacks, verbose=2)

    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, labels=range(len(encoder.classes_)),
                                target_names=encoder.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred, labels=range(len(encoder.classes_))))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(encoder, f)
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved label encoder to {ENCODER_OUT}")


if __name__ == "__main__":
    main()
