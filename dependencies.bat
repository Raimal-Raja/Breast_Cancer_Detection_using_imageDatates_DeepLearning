@echo off
REM Breast Cancer Detection - TensorFlow 2.16+ Fix (Keras 3)
REM This installs TensorFlow 2.16 which has Keras 3

echo =============================================================
echo BREAST CANCER DETECTION - TENSORFLOW 2.16 FIX (KERAS 3)
echo =============================================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo ERROR: Virtual environment not found!
    echo Please run this from: D:\BreastCancerDetection\flask_app
    pause
    exit /b 1
)

echo.
echo Step 1: Uninstalling ALL TensorFlow/Keras versions...
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel keras keras-nightly tf-keras -y

echo.
echo Step 2: Installing TensorFlow 2.16.1 (with Keras 3)...
pip install tensorflow==2.16.1

echo.
echo Step 3: Installing other dependencies...
pip install flask==3.0.0
pip install opencv-python==4.8.1.78
pip install pillow==10.1.0
pip install "numpy<2.0"

echo.
echo Step 4: Verifying installation...
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import keras; print(f'Keras: {keras.__version__}')"

echo.
echo =============================================================
echo Checking Keras version...
echo =============================================================
python -c "import keras; v = int(keras.__version__.split('.')[0]); print(f'Keras major version: {v}'); print('COMPATIBLE with your model!' if v >= 3 else 'ERROR: Still using Keras 2!')"

echo.
echo =============================================================
echo FIX COMPLETE!
echo =============================================================
echo.
echo Next steps:
echo   1. Run: python app.py
echo.
pause