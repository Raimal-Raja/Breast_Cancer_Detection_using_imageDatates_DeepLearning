# Breast Cancer Detection Using Deep Learning

An automated system for detecting **Invasive Ductal Carcinoma (IDC)** in breast histopathology images using EfficientNetB0 and transfer learning. The model is trained in Google Colab and deployed locally as a Flask web application.

---

## Project Overview

| Field | Details |
|---|---|
| **Author** | Raimal Raja Kolhi |
| **Supervisor** | Assistant Professor Dileep Kumar |
| **Institution** | University of Sindh, Laar Campus |
| **Date** | 31 January 2026 |
| **Dataset** | [Breast Histopathology Images](https://www.kaggle.com/datasets/janko/breast-histopathology-images) (277,524 images) |
| **Architecture** | EfficientNetB0 (Transfer Learning) |
| **Test Accuracy** | 78.95% |
| **AUC Score** | 0.8552 |

---

## How It Works

1. A histopathology image is uploaded through the web interface.
2. The Flask backend preprocesses the image — resizes it to 96×96, normalizes pixel values to [0, 1], and converts BGR to RGB via OpenCV.
3. The preprocessed image is fed into the trained EfficientNetB0 model.
4. The model outputs the probability of IDC being present, along with a confidence score.
5. Results are displayed on the dashboard instantly.

---

## Project Structure

```
BreastCancerDetection/
├── flask_app/
│   ├── app.py                  # Flask backend (routes, prediction logic)
│   ├── templates/
│   │   └── index.html          # Web interface
│   ├── static/
│   │   ├── css/style.css       # Stylesheet
│   │   └── js/script.js        # Frontend JavaScript
│   ├── models/
│   │   ├── breast_cancer_model.keras   # Trained model (from Colab)
│   │   └── model_config.pkl            # Model configuration & metrics
│   ├── uploads/                # Temporary upload directory (auto-created)
│   └── requirements.txt        # Python dependencies
└── breast_cancer_colab_training.ipynb  # Google Colab training notebook
```

---

## Setup & Installation

### Prerequisites

- Python 3.10
- A GPU is recommended for training but not required for running the Flask app.

### 1. Clone or Download the Project

Place the project folder on your local machine and navigate into `flask_app/`:

```bash
cd BreastCancerDetection/flask_app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Version |
|---|---|
| Python | 3.10.0 |
| TensorFlow | 2.16.1 |
| Keras | 3.0 |
| Flask | 3.0.0 |
| OpenCV | 4.8.1.78 |
| NumPy | 2.0 |

### 4. Add the Trained Model

Copy the following files from your Google Drive (saved during Colab training) into `flask_app/models/`:

- `breast_cancer_model.keras`
- `model_config.pkl`

### 5. Run the Application

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## Training (Google Colab)

The notebook `breast_cancer_colab_training.ipynb` handles the full training pipeline.

### Kaggle API Setup

1. Create a Kaggle account and go to **Settings → API**.
2. Generate a new token named `BreastCancerDetection` — this downloads `kaggle.json`.
3. Upload `kaggle.json` to your Colab environment.
4. The notebook uses the Kaggle API to download the dataset automatically.

### Training Approach

Training uses a **two-phase strategy**:

| Phase | Description |
|---|---|
| **Phase 1** | Train for 6 epochs with the EfficientNetB0 base frozen — only the custom classification head learns. |
| **Phase 2** | Unfreeze the last 30 layers and fine-tune with a reduced learning rate for deeper feature adaptation. |

### Data Augmentation

`ImageDataGenerator` applies the following transforms to improve generalization:

- Random rotation
- Horizontal and vertical flips
- Random zoom

### Model Architecture

```
EfficientNetB0 (pretrained, ImageNet)
        ↓
  Global Average Pooling
        ↓
  Batch Normalization
        ↓
  Dropout (0.5)
        ↓
  Dense Layer
        ↓
  Dropout (0.3)
        ↓
  Sigmoid Output (IDC probability)
```

### Google Drive Storage

Trained outputs are saved to Google Drive for persistence across Colab sessions:

| Path | Contents |
|---|---|
| `/models/` | `best_model.keras`, `model_config.pkl` |
| `/results/` | Confusion matrices, training history plots |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web dashboard |
| `/predict` | POST | Accepts an image file, returns IDC prediction as JSON |
| `/health` | GET | Returns model status and configuration |

### `/predict` Response Example

```json
{
  "predicted_class": 1,
  "class_name": "IDC Positive",
  "confidence": 87.34,
  "probability_positive": 87.34,
  "probability_negative": 12.66,
  "image": "data:image/png;base64,..."
}
```

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Test Accuracy | 78.95% | Achieved with 6 epochs of training |
| AUC Score | 0.8552 | Strong class discrimination |
| Target Accuracy | 90%+ | Achievable by increasing training to ~20 epochs |

---

## Disclaimer

This is a research project and an AI-assisted diagnostic tool. It is **not** intended for clinical use. All medical diagnosis and treatment decisions must be made by qualified healthcare professionals.