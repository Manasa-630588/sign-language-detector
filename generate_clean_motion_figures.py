import os, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

DATA_DIR = "data/sequences"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load only folders that have > 0 samples
X, y_raw = [], []
for label in sorted(os.listdir(DATA_DIR)):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir): continue
    files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    if len(files) == 0: continue  # Skip classes with 0 samples
    for f in sorted(files):
        X.append(np.load(os.path.join(label_dir, f)))
        y_raw.append(label)

X = np.array(X, dtype=np.float32)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
class_names = encoder.classes_
print(f"Loaded {len(X)} samples across {len(class_names)} trained classes: {list(class_names)}")

# Stratified split or safe split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Training clean motion model for figures...")
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

# Check present classes in validation
present_labels = np.unique(np.concatenate([y_val, y_pred]))
present_names = class_names[present_labels]

# Clean Motion Classification Report
report_df = pd.DataFrame(classification_report(y_val, y_pred, labels=present_labels, target_names=present_names, output_dict=True, zero_division=0)).iloc[:-1, :-3].T
plt.figure(figsize=(10, max(6, len(present_names) * 0.35)))
sns.heatmap(report_df, annot=True, cmap="Greens", fmt=".2f", cbar=True)
plt.title("Motion Model Classification Report (Trained Classes)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "clean_motion_classification_report.png"), dpi=300)
plt.close()

# Clean Motion Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title("Motion Model Confusion Matrix (Trained Classes)", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "clean_motion_confusion_matrix.png"), dpi=300)
plt.close()
print("Successfully generated clean motion figures without 0-sample classes!")
