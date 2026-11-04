from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# 1. Load your trained static model
print('Loading model...')
model = tf.keras.models.load_model('models/sign_classifier.h5')

# 2. Define the specific overlapping static classes
target_classes = ['b', 'd', 'e', 'g', 'i', 'k', 'n', 'r', 'v']

# 3. Load your static landmarks dataset safely
print('Loading dataset...')
df = pd.read_csv('data/landmarks.csv', header=None, low_memory=False)
df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()]

X = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values.astype(str)

# Clean and standardize string labels
y_clean = np.array([val.strip().lower() for val in y_raw])
target_lower = [c.lower() for c in target_classes]

# Filter dataset to strictly include target classes
mask = np.isin(y_clean, target_lower)
X_filtered = X[mask]
y_filtered = y_clean[mask]

print(
    f'Filtered down to {len(X_filtered)} samples across target classes for evaluation.'
)

# 4. Perform direct model inference
preds = model(X_filtered, training=False).numpy()
pred_indices = np.argmax(preds, axis=1)

# Map model predictions directly to our target classes using unique integer mapping or direct string matching
# Let's map predictions using the position in target_classes if model outputs correspond to them,
# or map via your classes list order.
# To keep it completely robust, let's map predictions based on the classes your dataset actually used:
unique_dataset_classes = sorted(list(set(y_clean)))
predicted_class_labels = np.array(
    [
        unique_dataset_classes[idx]
        if idx < len(unique_dataset_classes)
        else 'unknown'
        for idx in pred_indices
    ]
)

# Final clean evaluation arrays filtered only to target classes
final_mask = np.isin(y_filtered, target_lower) & np.isin(
    predicted_class_labels, target_lower
)
y_true_final = [val.upper() for val in y_filtered[final_mask]]
y_pred_final = [val.upper() for val in predicted_class_labels[final_mask]]
target_upper = [c.upper() for c in target_classes]

# 5. Generate Confusion Sub-Matrix
cm = confusion_matrix(y_true_final, y_pred_final, labels=target_upper)

# 6. Plotting the Sub-Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_upper,
    yticklabels=target_upper,
    linewidths=0.5,
    linecolor='white',
)

plt.xlabel('Predicted labels', fontsize=12, fontweight='bold')
plt.ylabel('True label', fontsize=12, fontweight='bold')
plt.title(
    'Static Model Confusion Sub-Matrix (Initial Letter Overlaps)',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# Save image for your project report
plt.savefig('static_confusion_submatrix.png', dpi=300)
print(
    "Success! Confusion sub-matrix generated and saved as 'static_confusion_submatrix.png'."
)
plt.show()