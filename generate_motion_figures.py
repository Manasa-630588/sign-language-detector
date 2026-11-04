import os, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

DATA_DIR = "data/sequences"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

X, y_raw = [], []
for label in sorted(os.listdir(DATA_DIR)):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir): continue
    for f in sorted(os.listdir(label_dir)):
        if f.endswith(".npy"):
            X.append(np.load(os.path.join(label_dir, f)))
            y_raw.append(label)

X = np.array(X, dtype=np.float32)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
class_names = encoder.classes_

n_val = max(1, int(round(0.2 * len(X))))
X_train, X_val = X[:-n_val], X[-n_val:]
y_train, y_val = y[:-n_val], y[-n_val:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Training motion model for figures...")
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

# Get unique labels present in validation set to avoid mismatch
present_labels = np.unique(np.concatenate([y_val, y_pred]))
present_names = class_names[present_labels]

# Motion Classification Report
report_df = pd.DataFrame(classification_report(y_val, y_pred, labels=present_labels, target_names=present_names, output_dict=True, zero_division=0)).iloc[:-1, :-3].T
plt.figure(figsize=(10, max(6, len(present_names) * 0.4)))
sns.heatmap(report_df, annot=True, cmap="Greens", fmt=".2f")
plt.title("Motion Model Classification Report", fontsize=14, pad=15)
plt.savefig(os.path.join(FIGURES_DIR, "motion_classification_report.png"), dpi=300)
plt.close()

# Motion Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Motion Model Confusion Matrix", fontsize=14, pad=15)
plt.savefig(os.path.join(FIGURES_DIR, "motion_confusion_matrix.png"), dpi=300)
plt.close()
print("Successfully generated motion figures!")
