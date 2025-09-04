
---

# Spoken Digit Recognition Project

This project implements a **spoken digit recognition system** using deep learning with PyTorch and torchaudio. It supports training, evaluation, and inference (both CLI and Streamlit app) for spoken digits `0-9`.

The project integrates **MLflow** and **DVC** for experiment tracking, versioning, and reproducibility, while loading datasets directly from **Hugging Face**, so no local dataset storage is required.

---

## Features

* Load the **Spoken Digit Dataset** directly from Hugging Face.
* Preprocessing pipeline:

  * Resample audio to a target sample rate.
  * Normalize waveform between -1 and 1.
  * Convert waveform to **log Mel spectrogram**.
  * Pad/truncate sequences to handle variable audio lengths.
* Trainable PyTorch model for digit classification.
* Experiment tracking and model versioning with **MLflow**.
* Dependency management using **uv**.
* Predict digits from audio files or in-memory bytes.
* Interactive **Streamlit** app for live prediction using microphone or file upload.
* End to end training and experiment `notebook` implementation

---

## Getting Started

### 1️⃣ Clone the Repository

```bash
git clone git@github.com:daunsid/audio2digit.git
cd audio2digit
```

---

### 2️⃣ Set Up Dependencies with `uv`

```bash
# Install uv if needed
pip install uv

# Sync environment
uv sync

# install project as package
uv pip install -e .
```

* Ensures reproducible Python environment.

---

### 3️⃣ Fetch Dataset from Hugging Face

* The dataset is automatically downloaded during training.
* No need for local storage or manual DVC data pull.

---

### 4️⃣ Preprocessing

* **Resampling:** Audio is resampled to the target sample rate (e.g., 8000 Hz).
* **Normalization:** Waveform values scaled to \[-1, 1].
* **Feature extraction:** Convert waveform to log Mel spectrogram.
* **Padding/truncating:** Handles variable-length audio sequences for batching.

---

### 5️⃣ Training

```bash
dvc repro
```

* Trains the model and logs metrics, parameters, and the trained model to **MLflow**.
* To view experiments:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open at `http://127.0.0.1:5000`.

---

### 6️⃣ Evaluation

```bash
uv run python -m ai.scripts.evaluation
```

* Computes accuracy, confusion matrix, and other evaluation metrics on test data.

---

### 7️⃣ Predict Digits (CLI or Script)

```python
from ai.predict import predict_digit

# From audio file
pred = predict_digit(model_path="model.pth", audio_path="sample.wav")
print("Predicted digit:", pred)

# From in-memory bytes
with open("sample.wav", "rb") as f:
    audio_bytes = f.read()

pred = predict_digit(model_path="model.pth", audio_byte=audio_bytes)
print("Predicted digit:", pred)
```

---

### 8️⃣ Streamlit App

```bash
uv run streamlit run src/app/app.py
```

* Upload audio file or record via microphone.
* The app predicts the spoken digit using the trained model.

---

### 9️⃣ Optional: Update Dependencies

```bash
# Add new dependency to requirements.in
uv compile
uv sync
```

* Keeps your environment consistent and reproducible.

---

<!-- ## Project Structure

```
├── app.py              # Streamlit app for inference
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── predict.py          # CLI inference functions
├── preprocess.py       # Audio preprocessing pipeline
├── models.py           # Model architecture
├── utils.py            # Helper functions
├── requirements.in     # uv dependencies
├── requirements.txt    # Generated dependencies
└── README.md
``` -->

---

### Notes

* **DVC**: Used for tracking experiment artifacts and datasets if needed in future.
* **MLflow**: Logs parameters, metrics, and model checkpoints for reproducibility.
* **Hugging Face**: Dataset is streamed dynamically, so no local storage required.

---
