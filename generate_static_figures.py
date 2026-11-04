import os, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

CSV_PATH = "data/landmarks.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
X, y = df.drop("label", axis=1).values, df["label"].values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Training model for figures...")
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

y_pred = np.argmax(model.predict(X_test), axis=1)

# Figure 4.3
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).iloc[:-1, :-3].T
plt.figure(figsize=(10, max(6, len(class_names) * 0.4)))
sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f")
plt.title("Figure 4.3: Static Model First Training Run Classification Report", fontsize=14, pad=15)
plt.savefig(os.path.join(FIGURES_DIR, "figure_4_3_classification_report.png"), dpi=300)
plt.close()

# Figure 4.4
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Figure 4.4: Static Model Confusion Matrix After First Training Run", fontsize=14, pad=15)
plt.savefig(os.path.join(FIGURES_DIR, "figure_4_4_confusion_matrix.png"), dpi=300)
plt.close()
print("Successfully generated and saved Figures 4.3 and 4.4!")
