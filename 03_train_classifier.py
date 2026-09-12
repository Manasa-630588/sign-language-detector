"""
Day 6-7: Train a baseline classifier on the collected landmark data.

Trains a simple feedforward neural network on the 63-dim landmark vectors.
Saves the trained model and label encoder for use in real-time inference.

Run with:
    python 03_train_classifier.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import os

DATA_CSV = "../data/landmarks.csv"
MODEL_OUT = "../models/sign_classifier.h5"
ENCODER_OUT = "../models/label_encoder.pkl"


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save model + encoder
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved label encoder to {ENCODER_OUT}")


if __name__ == "__main__":
    main()
