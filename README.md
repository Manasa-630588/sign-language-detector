# Real-Time Dual-Model Sign Language Detection System

An end-to-end, non-invasive sign language detection framework built with **MediaPipe HandLandmarker**, **TensorFlow/Keras**, and **Streamlit**. The system processes live webcam feeds to translate static fingerspelling alphabets, numeric digits, and dynamic two-handed conversational signs into continuous text without requiring specialized sensor gloves or GPU acceleration.

---

## Key Features

- **Dual-Model Architecture**:
  - **Static Alphabet & Digits Classifier**: Feedforward Artificial Neural Network (ANN) processing single-frame normalized 63-D keypoint vectors across 37 classes (`A–Z` and `0–10`). Achieves **97.8% validation accuracy**.
  - **Dynamic Motion Sequence Classifier**: Stacked two-layer Long Short-Term Memory (LSTM) network processing 30-frame temporal windows with 132-D dual-hand keypoints across 48 sequence classes (47 dynamic words/phrases/motion letters + 1 explicit `IDLE` non-gesture baseline). Achieves **94.2% validation accuracy**.
- **Spatial Invariance**: Keypoint normalization relative to wrist origin and Euclidean scaling eliminates sensitivity to lighting, camera distance (0.5–1.5m), and hand proportions.
- **Dual-Hand Support**: Tracks up to two hands simultaneously (`num_hands=2`). Single-handed signs automatically zero-pad secondary hand indices.
- **Interactive Streamlit Web Dashboard**: Features real-time webcam frame rendering, confidence threshold controls, temporal majority voting, hold accumulator timers, and sentence construction controls (`Space`, `Backspace`, `Clear`).
- **CPU-Optimized**: Runs natively at **15–21 FPS** on standard multi-core laptop CPUs.

---

## Repository Structure

```text
sign-language-detection/
├── data/
│   ├── landmarks.csv             # 37-class static landmark dataset (11,100 rows)
│   └── sequences/                # 48-class dynamic motion dataset (1,840 .npy files)
│       ├── HELLO/
│       ├── THANKS/
│       ├── IDLE/
│       └── ...
├── models/
│   ├── sign_classifier.h5        # Trained static ANN model
│   ├── label_encoder.pkl         # Static label encoder mappings
│   ├── sequence_classifier.h5    # Trained dynamic sequence LSTM model
│   └── seq_label_encoder.pkl     # Dynamic label encoder mappings
├── scripts/
│   ├── 01_test_hand_tracking.py  # MediaPipe webcam pipeline verification
│   ├── 02_collect_landmark_data.py # Static single-frame dataset recorder
│   ├── 03_train_classifier.py    # Feedforward ANN training & sub-matrix generator
│   ├── 04_collect_sequence_data.py # 30-frame dynamic sequence recorder (.npy)
│   ├── 05_train_sequence_model.py # 2-layer LSTM training with synthetic data augmentation
│   ├── submatrix.py              # Targeted static confusion sub-matrix evaluator
│   └── motion_slice.py           # Motion model sequence slice confusion matrix generator
├── app.py                        # Streamlit real-time interactive web application
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
```

---

## Environment Setup & Installation

> **Note**: Python 3.11.9 is strictly recommended due to C-extension binary compatibility issues with MediaPipe on newer Python releases.

1. **Clone the repository and navigate to the project directory**:
   ```bash
   git clone https://github.com/your-username/sign-language-detection.git
   cd sign-language-detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Hardware Dependencies
- **CPU**: Multi-core processor (Intel Core i5/i7 or AMD Ryzen equivalent).
- **Camera**: Standard integrated or USB RGB webcam (720p/1080p).
- **RAM**: 8 GB minimum (16 GB recommended for live Streamlit rendering).

---

## Execution Workflow

### Step 1: Verify Camera & MediaPipe Tracking
```bash
python scripts/01_test_hand_tracking.py
```
Confirms that MediaPipe HandLandmarker initializes properly and renders 21 3D joint landmark skeletons on your live camera feed.

### Step 2: Static Alphabet Pipeline (ANN)
1. **Data Collection**:
   ```bash
   python scripts/02_collect_landmark_data.py
   ```
   Hold each static hand shape in front of the camera and press `SPACE` to collect samples (300 samples per class recorded to `data/landmarks.csv`).
2. **Train Static Classifier**:
   ```bash
   python scripts/03_train_classifier.py
   ```
   Trains the feedforward ANN on the 63-D feature vectors and outputs `models/sign_classifier.h5`.

### Step 3: Motion Sequence Pipeline (LSTM)
1. **Sequence Data Collection**:
   ```bash
   python scripts/04_collect_sequence_data.py
   ```
   Captures 30-frame temporal sequences (shape: `30 x 132`) for dynamic signs (`J`, `Z`, `10`, `HELLO`, `THANKS`, `IDLE`, etc.) into `.npy` binary arrays under `data/sequences/`.
2. **Train Sequence Model with Augmentation**:
   ```bash
   python scripts/05_train_sequence_model.py
   ```
   Applies synthetic temporal warping (±12 frames time shift, 0.7x–1.5x speed scaling, Gaussian joint jitter) and trains the 2-layer LSTM model saved to `models/sequence_classifier.h5`.

### Step 4: Evaluate Model Matrices
Generate high-resolution confusion matrix figures for project reports:
```bash
# Static sub-matrix evaluation
python submatrix.py

# Motion sequence slice evaluation
python motion_slice.py
```

### Step 5: Run Interactive Streamlit Web App
Launch the real-time detection dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Use the sidebar controls to toggle between **Alphabet Mode** and **Motion Mode**, adjust confidence sliders, and build continuous sentences live!

---

## System Architecture Summary

| Component | Static Alphabet Model | Dynamic Motion Model |
| :--- | :--- | :--- |
| **Input Shape** | `(63,)` normalized coordinates | `(30, 132)` sequence tensor |
| **Hand Tracking** | 1 Hand (`num_hands=1`) | Up to 2 Hands (`num_hands=2`) |
| **Model Structure** | Feedforward ANN (Dense 128 → 64) | Stacked LSTM (LSTM 64 → 32 → Dense 32) |
| **Target Classes** | 37 (A–Z, 0–10) | 48 (47 dynamic signs + 1 IDLE class) |
| **Validation Acc.** | **97.8%** | **94.2%** (Chronological) / **91.6%** (Time-shifted) |
| **Latency** | ~15 ms per frame | ~30 ms sliding window |

---

## Authors & Acknowledgments

- **Team Members**: K. Bhashitha, M. Chinni, P. Manasa
- **Project Guide**: Dr. Y. Usha Rani (Associate Professor Grade-1)
- **Institution**: School of Computer Science and Engineering, VIT-AP University