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
MODEL_PATH = 'models/breast_cancer_model.h5'
CONFIG_PATH = 'models/model_config.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and configuration
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(CONFIG_PATH, 'rb') as f:
    model_config = pickle.load(f)

IMG_SIZE = model_config['img_size']
CLASS_NAMES = model_config['class_names']

print(f"Model configuration: {model_config}")

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    # Load image
    img = cv2.imread(image_path)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def get_prediction(image_path):
    """Get prediction from model"""
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction_proba = model.predict(img, verbose=0)[0][0]
    
    # Get predicted class
    predicted_class = 1 if prediction_proba > 0.5 else 0
    
    # Prepare result
    result = {
        'predicted_class': int(predicted_class),
        'class_name': CLASS_NAMES[predicted_class],
        'confidence': float(prediction_proba) if predicted_class == 1 else float(1 - prediction_proba),
        'probability_positive': float(prediction_proba),
        'probability_negative': float(1 - prediction_proba)
    }
    
    return result

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html', 
                         model_accuracy=f"{model_config['test_accuracy']*100:.2f}",
                         model_auc=f"{model_config['test_auc']:.4f}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save uploaded file
            filename = 'temp_image.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            result = get_prediction(filepath)
            
            # Read image and encode as base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            result['image'] = f"data:image/png;base64,{img_data}"
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'img_size': IMG_SIZE
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("BREAST CANCER DETECTION - FLASK APPLICATION")
    print("="*70)
    print(f"Model Accuracy: {model_config['test_accuracy']*100:.2f}%")
    print(f"Model AUC: {model_config['test_auc']:.4f}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("="*70)
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)