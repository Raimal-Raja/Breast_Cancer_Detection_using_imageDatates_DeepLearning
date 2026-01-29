# Breast Cancer Detection - TensorFlow 2.16+ Fix (Keras 3)
# Run in PowerShell: .\fix_tensorflow_v2.ps1

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "BREAST CANCER DETECTION - TENSORFLOW 2.16 FIX (KERAS 3)" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "No virtual environment detected" -ForegroundColor Yellow
    Write-Host "Attempting to activate .venv..." -ForegroundColor Yellow

    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & .venv\Scripts\Activate.ps1
        Write-Host "Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "Virtual environment not found!" -ForegroundColor Red
        Write-Host "Run this script from: D:\BreastCancerDetection\flask_app" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Step 1: Uninstalling ALL TensorFlow/Keras versions..." -ForegroundColor Yellow
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel keras keras-nightly tf-keras -y

Write-Host ""
Write-Host "Step 2: Installing TensorFlow 2.16.1 (with Keras 3)..." -ForegroundColor Yellow
pip install tensorflow==2.16.1

Write-Host ""
Write-Host "Step 3: Installing other dependencies..." -ForegroundColor Yellow
pip install flask==3.0.0
pip install opencv-python==4.8.1.78
pip install pillow==10.1.0
pip install "numpy<2.0"

Write-Host ""
Write-Host "Step 4: Verifying installation..." -ForegroundColor Yellow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras; print('Keras:', keras.__version__)"

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "Checking Keras version..." -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

$kerasVersion = python -c "import keras; print(keras.__version__)"
$major = $kerasVersion.Split('.')[0]

Write-Host "Keras major version: $major"

if ([int]$major -ge 3) {
    Write-Host "COMPATIBLE with your model!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Still using Keras 2!" -ForegroundColor Red
}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "FIX COMPLETE!" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: python app.py" -ForegroundColor White
