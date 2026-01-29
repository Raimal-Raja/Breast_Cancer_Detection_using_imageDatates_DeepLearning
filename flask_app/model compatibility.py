"""
BREAST CANCER DETECTION - SYSTEM DIAGNOSTIC
Run this script to check if everything is set up correctly
"""

import os
import sys

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_python():
    print_header("PYTHON VERSION")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

def check_dependencies():
    print_header("CHECKING DEPENDENCIES")
    
    dependencies = {
        'tensorflow': 'TensorFlow',
        'flask': 'Flask',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:15} - Version: {version}")
        except ImportError:
            print(f"✗ {name:15} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install tensorflow flask opencv-python pillow numpy")
    else:
        print("\n✓ All dependencies installed!")

def check_tensorflow():
    print_header("TENSORFLOW DETAILS")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"Keras Version: {tf.keras.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU Available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("⚠ No GPU detected (CPU will be used)")
        
        # Check if model can be loaded
        print("\nTesting model loading capability...")
        from tensorflow import keras
        print("✓ Keras import successful")
        
    except ImportError:
        print("✗ TensorFlow not installed")

def check_file_structure():
    print_header("CHECKING FILE STRUCTURE")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # Define expected structure
    required_files = {
        'app.py': 'Main Flask application',
        'models/breast_cancer_model.keras': 'Trained model (should be ~33 MB)',
        'models/model_config.pkl': 'Model configuration',
        'templates/index.html': 'Web interface'
    }
    
    optional_files = {
        'static/css/style.css': 'Styles',
        'static/js/script.js': 'JavaScript',
        'uploads/': 'Upload directory'
    }
    
    print("\nRequired files:")
    all_found = True
    
    for file_path, description in required_files.items():
        full_path = os.path.join(script_dir, file_path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.2f} MB"
                else:
                    size_str = f"{size/1024:.2f} KB"
                print(f"✓ {file_path:40} ({size_str}) - {description}")
            else:
                print(f"✓ {file_path:40} (directory) - {description}")
        else:
            print(f"✗ {file_path:40} MISSING - {description}")
            all_found = False
    
    print("\nOptional files:")
    for file_path, description in optional_files.items():
        full_path = os.path.join(script_dir, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path:40} - {description}")
        else:
            print(f"⚠ {file_path:40} - {description}")
    
    if not all_found:
        print("\n✗ Some required files are missing!")
        print("Please ensure all files from Google Drive are in the correct locations.")
    else:
        print("\n✓ All required files found!")

def test_model_loading():
    print_header("TESTING MODEL LOADING")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models', 'breast_cancer_model.keras')
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return
        
        print(f"Model path: {model_path}")
        print(f"Model size: {os.path.getsize(model_path)/(1024*1024):.2f} MB")
        
        print("\nAttempting to load model...")
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        print("✓ Model loaded successfully!")
        
        print(f"\nModel details:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")
        
        # Test prediction
        import numpy as np
        test_input = np.random.random((1, 96, 96, 3)).astype(np.float32)
        print("\nTesting prediction...")
        prediction = model.predict(test_input, verbose=0)
        print(f"✓ Test prediction successful!")
        print(f"  Output: {prediction[0][0]:.4f}")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()

def check_network():
    print_header("CHECKING NETWORK")
    
    import socket
    
    # Check if port 5000 is available
    port = 5000
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"⚠ Port {port} is already in use")
            print("  Another application might be using this port")
            print("  You can change the port in app.py")
        else:
            print(f"✓ Port {port} is available")
    except Exception as e:
        print(f"⚠ Could not check port: {e}")

def print_summary():
    print_header("DIAGNOSTIC SUMMARY")
    
    print("""
If all checks passed (✓), you're ready to run the application:

1. Open terminal/command prompt
2. Navigate to the flask_app directory:
   cd D:\\BreastCancerDetection\\flask_app

3. Run the application:
   python app.py

4. Open your browser:
   http://127.0.0.1:5000

If any checks failed (✗), please address those issues first.
Refer to TROUBLESHOOTING.md for detailed solutions.
    """)

def main():
    print("="*70)
    print("BREAST CANCER DETECTION - DIAGNOSTIC TOOL")
    print("="*70)
    
    check_python()
    check_dependencies()
    check_tensorflow()
    check_file_structure()
    test_model_loading()
    check_network()
    print_summary()

if __name__ == '__main__':
    main()