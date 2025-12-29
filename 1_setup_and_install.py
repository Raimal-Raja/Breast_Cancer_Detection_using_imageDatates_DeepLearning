"""
Breast Cancer Detection System - Google Colab Setup
Run this FIRST in Google Colab to set up everything
"""

import os
import sys

print("=" * 80)
print("BREAST CANCER DETECTION SYSTEM - GOOGLE COLAB SETUP")
print("=" * 80)

# Step 1: Install all dependencies
print("\n[1/6] Installing core dependencies...")
!pip install -q kagglehub pandas numpy pillow opencv-python matplotlib seaborn scikit-learn tqdm

print("\n[2/6] Installing TensorFlow...")
!pip install -q tensorflow

print("\n[3/6] Installing visualization tools...")
!pip install -q plotly

print("\n[4/6] Installing Streamlit and ngrok...")
!pip install -q streamlit pyngrok streamlit-option-menu

print("\n[5/6] Installing additional utilities...")
!pip install -q pyyaml

print("\n✅ All dependencies installed successfully!")

# Step 2: Create directory structure
print("\n[6/6] Creating project directory structure...")
directories = [
    'data/raw',
    'data/processed',
    'data/training/benign',
    'data/training/malignant',
    'data/validation/benign',
    'data/validation/malignant',
    'data/testing/benign',
    'data/testing/malignant',
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

print("✅ Project structure created successfully!")

# Step 3: Verify imports
print("\n[VERIFICATION] Checking imports...")
try:
    import tensorflow as tf
    import cv2
    import pandas as pd
    import numpy as np
    from PIL import Image
    import kagglehub
    import streamlit as st
    from pyngrok import ngrok
    
    print(f"  ✅ TensorFlow version: {tf.__version__}")
    print(f"  ✅ OpenCV version: {cv2.__version__}")
    print(f"  ✅ NumPy version: {np.__version__}")
    print(f"  ✅ Pandas version: {pd.__version__}")
    print("  ✅ All libraries imported successfully!")
    
except ImportError as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Step 4: Set up ngrok for Streamlit
print("\n[NGROK SETUP] Setting up tunnel for Streamlit...")
print("Note: You'll need to provide your ngrok auth token later")
print("Get your free token at: https://dashboard.ngrok.com/get-started/your-authtoken")

print("\n" + "=" * 80)
print("SETUP COMPLETE! ✅")
print("=" * 80)
print("\n📋 NEXT STEPS:")
print("1. Run '2_data_loading.py' to load and prepare the dataset")
print("2. Run '3_model_training.py' to train the model")
print("3. Run '4_model_evaluation.py' to evaluate performance")
print("4. Run '5_launch_app.py' to start the Streamlit dashboard")
print("\n⚠️  IMPORTANT: Run cells in order!")
print("=" * 80)