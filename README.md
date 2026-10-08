# 🩺 Breast Cancer Detection Using Deep Learning

<div align="center">

![Breast Cancer Detection Banner](https://img.shields.io/badge/Deep%20Learning-EfficientNetB0-blue?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10-green?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An automated system for detecting Invasive Ductal Carcinoma (IDC) in breast histopathology images using EfficientNetB0 and Transfer Learning — trained on Google Colab and deployed as a Flask web application.**

[📋 Project Report](./Breast_Cancer_Detection_Project_Report.docx) • [🚀 Quick Start](#-setup--installation) • [📊 Results](#-results) • [🌐 API Reference](#-api-endpoints)

</div>

---

## 📸 Dashboard Preview

![Project Dashboard](https://github.com/Raimal-Raja/Breast_Cancer_Detection_using_imageDatates_DeepLearning/blob/main/Media/image.png?raw=true)

---

## 📌 Project Overview

| Field | Details |
|-------|---------|
| **Author** | Raimal Raja Kolhi |
| **Supervisor** | Prof Dileep Kumar |
| **Institution** | University of Sindh, Laar Campus |
| **Date** | 31 January 2026 |
| **Domain** | Medical Imaging & Computer Vision |
| **Dataset** | [Breast Histopathology Images](https://www.kaggle.com/datasets/janko/breast-histopathology-images) — 277,524 images |
| **Architecture** | EfficientNetB0 (Transfer Learning) |
| **Test Accuracy** | 78.95% |
| **AUC Score** | 0.8552 |
| **Target Accuracy** | 90%+ (achievable with ~20 epochs) |

---

## 🧠 Problem Statement

Manual diagnosis of breast cancer from histopathology slides is **time-consuming**, **labor-intensive**, and **prone to human error**. Traditional pathology requires specialists to examine tissue samples under microscopes — a process subject to inter-observer variability. This project tackles that challenge by building an automated, consistent, and rapid diagnostic assistant that classifies tissue patches as **IDC Positive** or **IDC Negative**.

---

## ⚙️ How It Works

```
User uploads histopathology image
           ↓
Flask backend receives the image
           ↓
OpenCV preprocessing: resize → 96×96, BGR→RGB, normalize [0,1]
           ↓
Image passed to trained EfficientNetB0 model
           ↓
Sigmoid output → IDC probability & confidence score
           ↓
Result displayed instantly on the dashboard
```

---

## 🏗️ Model Architecture

```
EfficientNetB0 (pretrained on ImageNet)
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
  Sigmoid Output → IDC Probability
```

EfficientNetB0 was selected for its excellent balance between model complexity and performance, making it well-suited for deployment in resource-constrained environments.

---

## 🗂️ Project Structure

```
BreastCancerDetection/
│
├── flask_app/
│   ├── app.py                          # Flask backend — routes & prediction logic
│   ├── templates/
│   │   └── index.html                  # Main web interface
│   ├── static/
│   │   ├── css/style.css               # Stylesheet
│   │   └── js/script.js                # Frontend JavaScript
│   ├── models/
│   │   ├── breast_cancer_model.keras   # Trained model (copied from Colab)
│   │   └── model_config.pkl            # Model configuration & metrics
│   ├── uploads/                        # Temporary upload directory (auto-created)
│   └── requirements.txt                # Python dependencies
│
├── breast_cancer_colab_training.ipynb  # Google Colab training notebook
├── requirements.txt                    # Root-level dependencies
├── requirements.ps1                    # PowerShell setup script (Windows)
├── Breast_Cancer_Detection_Project_Report.docx
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python **3.10**
- GPU recommended for training — not required for inference / running the Flask app

### 1. Clone the Repository

```bash
git clone https://github.com/Raimal-Raja/Breast_Cancer_Detection_using_imageDatates_DeepLearning.git
cd Breast_Cancer_Detection_using_imageDatates_DeepLearning/flask_app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

| OS | Command |
|----|---------|
| **Windows** | `venv\Scripts\activate` |
| **macOS / Linux** | `source venv/bin/activate` |

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the PowerShell script on Windows:

```powershell
.\requirements.ps1
```

### 4. Add the Trained Model Files

After training in Google Colab (see [Training](#-training-google-colab) below), copy the following files into `flask_app/models/`:

- `breast_cancer_model.keras`
- `model_config.pkl`

### 5. Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## ☁️ Training (Google Colab)

The notebook `breast_cancer_colab_training.ipynb` handles the complete training pipeline.

### Kaggle API Setup

1. Create a [Kaggle](https://www.kaggle.com) account and go to **Settings → API**.
2. Generate a new token — Kaggle will download a `kaggle.json` file.
3. Upload `kaggle.json` to your Colab session.
4. The notebook uses the Kaggle API to automatically download the dataset — no manual downloading needed.

### Two-Phase Training Strategy

| Phase | Layers | Epochs | Description |
|-------|--------|--------|-------------|
| **Phase 1** | Base frozen | 6 | Only the custom classification head is trained |
| **Phase 2** | Last 30 layers unfrozen | Fine-tune | Reduced learning rate for deeper feature adaptation |

### Data Augmentation

`ImageDataGenerator` applies the following transforms to improve generalization and reduce overfitting:

- Random rotation
- Horizontal and vertical flips
- Random zoom

### Google Drive Storage

Trained outputs are automatically saved to Google Drive for persistence across Colab sessions:

| Google Drive Path | Contents |
|-------------------|----------|
| `/models/` | `best_model.keras`, `model_config.pkl` |
| `/results/` | Confusion matrices, training history plots |

---

## 📊 Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 78.95% | Achieved with 6 epochs of training |
| **AUC Score** | 0.8552 | Strong discriminative ability between classes |
| **Target Accuracy** | 90%+ | Achievable by increasing training to ~20 epochs |

> The AUC score of 0.8552 confirms the model's strong ability to discriminate between IDC-positive and IDC-negative tissue samples, even at the initial training stage.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Serves the main web dashboard |
| `/predict` | `POST` | Accepts an image file, returns IDC prediction as JSON |
| `/health` | `GET` | Returns model status and configuration info |

### `/predict` — Request

Send a `multipart/form-data` POST request with a `file` field containing the histopathology image.

### `/predict` — Response Example

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

## 📦 Component Versions

| Component | Version |
|-----------|---------|
| Python | 3.10.0 |
| TensorFlow | 2.16.1 |
| Keras | 3.0 |
| Flask | 3.0.0 |
| OpenCV | 4.8.1.78 |
| NumPy | 2.0 |

---

## 🗺️ Development Lifecycle

```
Phase 1 — Kaggle API Integration
      ↓
Phase 2 — Cloud Training (Google Colab + GPU)
      ↓
Phase 3 — Local Deployment (Flask Web App)
```

This three-phase approach ensures efficient development, reproducibility, and scalability.

---

## 📁 Dataset

- **Name:** Breast Histopathology Images
- **Source:** [Kaggle](https://www.kaggle.com/datasets/janko/breast-histopathology-images)
- **Size:** 277,524 image patches
- **Task:** Binary classification — IDC Positive vs. IDC Negative
- **Image Size:** 50×50 pixels (patches extracted from whole-slide images; resized to 96×96 during preprocessing)

---

## ⚠️ Disclaimer

> This is a **research and educational project**. It is an AI-assisted diagnostic tool and is **not intended for clinical use**. All medical diagnosis and treatment decisions must be made by qualified healthcare professionals. Do not use model outputs as a substitute for professional medical advice.

---

## 👤 Author

**Raimal Raja Kolhi**
University of Sindh, Laar Campus
Supervised by: Assistant Professor Dileep Kumar

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com) — for the Breast Histopathology Images dataset
- [Google Colab](https://colab.research.google.com) — for free GPU-accelerated training environment
- [TensorFlow / Keras](https://www.tensorflow.org) — deep learning framework
- The medical imaging and open-source ML communities

---

<div align="center">
  <sub>Built with ❤️ for early cancer detection research</sub>
</div>

---

## Repository guide

### Contents

- [Breast_Cancer_Detection_Project_Report.docx](Breast_Cancer_Detection_Project_Report.docx)
- [Media](Media)
- [Project Report.docx](Project%20Report.docx)
- [Project Report.pdf](Project%20Report.pdf)
- [README.md](README.md)
- [breast_cancer_colab_training.ipynb](breast_cancer_colab_training.ipynb)
- [flask_app](flask_app)
- [requirements.ps1](requirements.ps1)
- [requirements.txt](requirements.txt)

### Getting started

```bash
git clone https://github.com/Raimal-Raja/Breast_Cancer_Detection_using_imageDatates_DeepLearning.git
cd Breast_Cancer_Detection_using_imageDatates_DeepLearning
```

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r "requirements.txt"
```

Application entry point:

```bash
python flask_app/app.py
```

### Configuration and limitations

Training needs the original image dataset and suitable compute. The Flask demo needs compatible saved model files. The project is an educational classifier.

### Validation

Reviewed on 2026-10-08. Python syntax checks passed for 1 source files. Syntax validation does not establish runtime correctness or dependency compatibility.

### Contributions

Describe the issue, reproduction steps, environment, and expected behavior when proposing a change. Keep generated environments, credentials, and unnecessary build artifacts out of new commits.

### License

No top-level license file was found during this review.
