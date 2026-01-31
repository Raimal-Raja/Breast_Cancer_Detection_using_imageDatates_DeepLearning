# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pickle
# import os
# from PIL import Image
# import cv2
# import base64
# from io import BytesIO

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# # MODEL_PATH = 'models/breast_cancer_model.h5'
# # CONFIG_PATH = 'models/model_config.pkl'

# MODEL_PATH = 'models/breast_cancer_model.h5'
# CONFIG_PATH = 'models/model_config.pkl'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# # Load model and configuration
# print("="*70)
# print("LOADING BREAST CANCER DETECTION MODEL")
# print("="*70)

# try:
#     model = keras.models.load_model(MODEL_PATH)
#     print("✓ Model loaded successfully!")
# except Exception as e:
#     print(f"✗ Error loading model: {e}")
#     print("\nMake sure you have:")
#     print("1. Trained the model on Google Colab")
#     print("2. Downloaded breast_cancer_model.h5")
#     print("3. Placed it in the 'models/' directory")
#     raise

# try:
#     with open(CONFIG_PATH, 'rb') as f:
#         model_config = pickle.load(f)
#     print("✓ Configuration loaded successfully!")
# except Exception as e:
#     print(f"✗ Error loading config: {e}")
#     raise

# IMG_SIZE = model_config['img_size']
# CLASS_NAMES = model_config['class_names']

# print(f"\nModel Configuration:")
# print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
# print(f"  Test Accuracy: {model_config['test_accuracy']*100:.2f}%")
# print(f"  Test AUC: {model_config['test_auc']:.4f}")
# print(f"  Model Type: {model_config.get('model_type', 'unknown').upper()}")
# print("="*70)

# def preprocess_image(image_path):
#     """Preprocess image for prediction"""
#     # Load image
#     img = cv2.imread(image_path)
    
#     if img is None:
#         raise ValueError("Unable to load image")
    
#     # Convert BGR to RGB
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Resize to model input size
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
#     # Normalize
#     img = img.astype(np.float32) / 255.0
    
#     # Add batch dimension
#     img = np.expand_dims(img, axis=0)
    
#     return img

# def get_prediction(image_path):
#     """Get prediction from model"""
#     # Preprocess image
#     img = preprocess_image(image_path)
    
#     # Make prediction
#     prediction_proba = model.predict(img, verbose=0)[0][0]
    
#     # Get predicted class
#     predicted_class = 1 if prediction_proba > 0.5 else 0
    
#     # Calculate confidence
#     confidence = prediction_proba if predicted_class == 1 else (1 - prediction_proba)
    
#     # Prepare result
#     result = {
#         'predicted_class': int(predicted_class),
#         'class_name': CLASS_NAMES[predicted_class],
#         'confidence': float(confidence * 100),  # Convert to percentage
#         'probability_positive': float(prediction_proba * 100),
#         'probability_negative': float((1 - prediction_proba) * 100)
#     }
    
#     return result

# @app.route('/')
# def index():
#     """Render home page"""
#     return render_template(
#         'index.html',
#         model_accuracy=f"{model_config['test_accuracy']*100:.2f}",
#         model_auc=f"{model_config['test_auc']:.4f}",
#         model_precision=f"{model_config.get('test_precision', 0):.4f}",
#         model_recall=f"{model_config.get('test_recall', 0):.4f}"
#     )

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle prediction request"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     # Check file extension
#     allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
#     file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
#     if file_ext not in allowed_extensions:
#         return jsonify({
#             'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
#         }), 400
    
#     try:
#         # Save uploaded file
#         filename = f'temp_image.{file_ext}'
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Get prediction
#         result = get_prediction(filepath)
        
#         # Read image and encode as base64 for display
#         with open(filepath, 'rb') as img_file:
#             img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
#         result['image'] = f"data:image/{file_ext};base64,{img_data}"
        
#         # Clean up
#         try:
#             os.remove(filepath)
#         except:
#             pass
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'img_size': IMG_SIZE,
#         'model_accuracy': f"{model_config['test_accuracy']*100:.2f}%"
#     })

# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("BREAST CANCER DETECTION - FLASK APPLICATION")
#     print("="*70)
#     print(f"Model Accuracy: {model_config['test_accuracy']*100:.2f}%")
#     print(f"Model AUC: {model_config['test_auc']:.4f}")
#     print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
#     print("="*70)
#     print("\nStarting Flask server...")
#     print("Open http://127.0.0.1:5000 in your browser")
#     print("="*70 + "\n")
    
#     app.run(debug=True, host='0.0.0.0', port=5000)




# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pickle
# import os
# from PIL import Image
# import cv2
# import base64
# from io import BytesIO

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'

# # FIXED: Changed to use .keras file instead of .h5
# MODEL_PATH = r'D:\BreastCancerDetection\flask_app\models\breast_cancer_model.keras'
# CONFIG_PATH = 'models/model_config.pkl'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# # Load model and configuration
# print("="*70)
# print("LOADING BREAST CANCER DETECTION MODEL")
# print("="*70)

# try:
#     model = keras.models.load_model(MODEL_PATH)
#     print("✓ Model loaded successfully!")
# except Exception as e:
#     print(f"✗ Error loading model: {e}")
#     print("\nMake sure you have:")
#     print("1. Trained the model on Google Colab")
#     print("2. Downloaded breast_cancer_model.keras (or .h5)")
#     print("3. Placed it in the 'models/' directory")
#     raise

# try:
#     with open(CONFIG_PATH, 'rb') as f:
#         model_config = pickle.load(f)
#     print("✓ Configuration loaded successfully!")
# except Exception as e:
#     print(f"✗ Error loading config: {e}")
#     raise

# IMG_SIZE = model_config['img_size']
# CLASS_NAMES = model_config['class_names']

# print(f"\nModel Configuration:")
# print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
# print(f"  Test Accuracy: {model_config['test_accuracy']*100:.2f}%")
# print(f"  Test AUC: {model_config['test_auc']:.4f}")
# print(f"  Model Type: {model_config.get('model_type', 'unknown').upper()}")
# print("="*70)

# def preprocess_image(image_path):
#     """Preprocess image for prediction"""
#     # Load image
#     img = cv2.imread(image_path)
    
#     if img is None:
#         raise ValueError("Unable to load image")
    
#     # Convert BGR to RGB
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Resize to model input size
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
#     # Normalize
#     img = img.astype(np.float32) / 255.0
    
#     # Add batch dimension
#     img = np.expand_dims(img, axis=0)
    
#     return img

# def get_prediction(image_path):
#     """Get prediction from model"""
#     # Preprocess image
#     img = preprocess_image(image_path)
    
#     # Make prediction
#     prediction_proba = model.predict(img, verbose=0)[0][0]
    
#     # Get predicted class
#     predicted_class = 1 if prediction_proba > 0.5 else 0
    
#     # Calculate confidence
#     confidence = prediction_proba if predicted_class == 1 else (1 - prediction_proba)
    
#     # Prepare result
#     result = {
#         'predicted_class': int(predicted_class),
#         'class_name': CLASS_NAMES[predicted_class],
#         'confidence': float(confidence * 100),  # Convert to percentage
#         'probability_positive': float(prediction_proba * 100),
#         'probability_negative': float((1 - prediction_proba) * 100)
#     }
    
#     return result

# @app.route('/')
# def index():
#     """Render home page"""
#     return render_template(
#         'index.html',
#         model_accuracy=f"{model_config['test_accuracy']*100:.2f}",
#         model_auc=f"{model_config['test_auc']:.4f}",
#         model_precision=f"{model_config.get('test_precision', 0):.4f}",
#         model_recall=f"{model_config.get('test_recall', 0):.4f}"
#     )

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle prediction request"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     # Check file extension
#     allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
#     file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
#     if file_ext not in allowed_extensions:
#         return jsonify({
#             'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
#         }), 400
    
#     try:
#         # Save uploaded file
#         filename = f'temp_image.{file_ext}'
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Get prediction
#         result = get_prediction(filepath)
        
#         # Read image and encode as base64 for display
#         with open(filepath, 'rb') as img_file:
#             img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
#         result['image'] = f"data:image/{file_ext};base64,{img_data}"
        
#         # Clean up
#         try:
#             os.remove(filepath)
#         except:
#             pass
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'img_size': IMG_SIZE,
#         'model_accuracy': f"{model_config['test_accuracy']*100:.2f}%"
#     })

# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("BREAST CANCER DETECTION - FLASK APPLICATION")
#     print("="*70)
#     print(f"Model Accuracy: {model_config['test_accuracy']*100:.2f}%")
#     print(f"Model AUC: {model_config['test_auc']:.4f}")
#     print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
#     print("="*70)
#     print("\nStarting Flask server...")
#     print("Open http://127.0.0.1:5000 in your browser")
#     print("="*70 + "\n")
    
#     app.run(debug=True, host='0.0.0.0', port=5000)




from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
from PIL import Image
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'

# FIXED: Use relative path and os.path for better compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'breast_cancer_model.keras')
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'model_config.pkl')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model and configuration
print("="*70)
print("LOADING BREAST CANCER DETECTION MODEL")
print("="*70)
print(f"Python working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

try:
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    
    # Load model with safe_mode=False to bypass version checks
    model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    print("✓ Model loaded successfully!")
    
    # Recompile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    print("✓ Model compiled successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify the model file exists:")
    print(f"   {MODEL_PATH}")
    print("2. Check TensorFlow version compatibility")
    print(f"   Current TensorFlow version: {tf.__version__}")
    print("3. Try updating TensorFlow:")
    print("   pip install --upgrade tensorflow")
    print("4. If using GPU, ensure CUDA/cuDNN are compatible")
    print("\nAvailable files in models directory:")
    if os.path.exists(os.path.join(BASE_DIR, 'models')):
        for file in os.listdir(os.path.join(BASE_DIR, 'models')):
            print(f"   - {file}")
    raise

try:
    print(f"\nLoading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'rb') as f:
        model_config = pickle.load(f)
    print("✓ Configuration loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load config file: {e}")
    print("Using default configuration...")
    # Fallback configuration
    model_config = {
        'img_size': 96,
        'class_names': ['No IDC', 'IDC Positive'],
        'test_accuracy': 0.7895,
        'test_auc': 0.8552,
        'test_precision': 0.56,
        'test_recall': 0.6364,
        'model_type': 'efficientnetb0'
    }

IMG_SIZE = model_config['img_size']
CLASS_NAMES = model_config['class_names']

print(f"\nModel Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Class Names: {CLASS_NAMES}")
print(f"  Test Accuracy: {model_config['test_accuracy']*100:.2f}%")
print(f"  Test AUC: {model_config['test_auc']:.4f}")
print(f"  Test Precision: {model_config.get('test_precision', 0):.4f}")
print(f"  Test Recall: {model_config.get('test_recall', 0):.4f}")
print(f"  Model Type: {model_config.get('model_type', 'unknown').upper()}")
print(f"  TensorFlow Version: {tf.__version__}")
print("="*70)

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Load image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Unable to load image from: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def get_prediction(image_path):
    """Get prediction from model"""
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Make prediction
        prediction_proba = model.predict(img, verbose=0)[0][0]
        
        # Get predicted class
        predicted_class = 1 if prediction_proba > 0.5 else 0
        
        # Calculate confidence
        confidence = prediction_proba if predicted_class == 1 else (1 - prediction_proba)
        
        # Prepare result
        result = {
            'predicted_class': int(predicted_class),
            'class_name': CLASS_NAMES[predicted_class],
            'confidence': float(confidence * 100),
            'probability_positive': float(prediction_proba * 100),
            'probability_negative': float((1 - prediction_proba) * 100)
        }
        
        return result
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

@app.route('/')
def index():
    """Render home page"""
    return render_template(
        'index.html',
        model_accuracy=f"{model_config['test_accuracy']*100:.2f}",
        model_auc=f"{model_config['test_auc']:.4f}",
        model_precision=f"{model_config.get('test_precision', 0)*100:.2f}",
        model_recall=f"{model_config.get('test_recall', 0)*100:.2f}"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Save uploaded file
        filename = f'temp_image_{os.getpid()}.{file_ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Get prediction
        result = get_prediction(filepath)
        
        # Read image and encode as base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        result['image'] = f"data:image/{file_ext};base64,{img_data}"
        
        # Clean up
        try:
            os.remove(filepath)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temporary file: {cleanup_error}")
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'img_size': IMG_SIZE,
        'model_accuracy': f"{model_config['test_accuracy']*100:.2f}%",
        'tensorflow_version': tf.__version__,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("BREAST CANCER DETECTION - FLASK APPLICATION")
    print("="*70)
    print(f"Model Accuracy: {model_config['test_accuracy']*100:.2f}%")
    print(f"Model AUC: {model_config['test_auc']:.4f}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("="*70)
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)