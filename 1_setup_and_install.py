"""
Breast Cancer Detection System - Setup and Installation
Run this first in Google Colab to install all dependencies
"""

import os
import sys

print("=" * 60)
print("BREAST CANCER DETECTION SYSTEM - SETUP")
print("=" * 60)

# Install required packages
print("\n[1/5] Installing core dependencies...")
!pip install -q kagglehub pandas numpy pillow opencv-python-headless matplotlib seaborn scikit-learn

print("\n[2/5] Installing deep learning frameworks...")
!pip install -q tensorflow==2.15.0

print("\n[3/5] Installing AutoML and visualization tools...")
!pip install -q pycaret plotly

print("\n[4/5] Installing Streamlit for web interface...")
!pip install -q streamlit streamlit-option-menu pyngrok

print("\n[5/5] Installing AWS SDK (optional for deployment)...")
!pip install -q boto3

print("\n✓ All dependencies installed successfully!")

# Create directory structure
print("\n[SETUP] Creating project directory structure...")
directories = [
    'data/raw',
    'data/processed',
    'data/training',
    'data/validation',
    'data/testing',
    'data/user_uploads',
    'models',
    'models/retraining_queue',
    'src',
    'app',
    'logs',
    'results'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"  ✓ Created: {directory}/")

print("\n✓ Project structure created successfully!")

# Import check
print("\n[VERIFICATION] Checking imports...")
try:
    import tensorflow as tf
    import cv2
    import pandas as pd
    import numpy as np
    from PIL import Image
    import kagglehub
    print("  ✓ TensorFlow version:", tf.__version__)
    print("  ✓ All core libraries imported successfully!")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SETUP COMPLETE! Ready to load dataset.")
print("=" * 60)
print("\nNext steps:")
print("1. Run '2_data_preprocessing.py' to load and preprocess data")
print("2. Run '3_automl_training.py' to train the model")
print("3. Run '4_model_evaluation.py' to evaluate performance")
print("4. Run '5_prediction.py' to test predictions")
print("5. Run '6_streamlit_app.py' to launch web interface")