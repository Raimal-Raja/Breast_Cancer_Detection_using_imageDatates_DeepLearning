"""
Complete Setup Script - Run This BEFORE Launching Streamlit
Ensures everything is properly configured
"""

import os
import sys

print("=" * 80)
print("BREAST CANCER DETECTION - COMPLETE SETUP")
print("=" * 80)

# Step 1: Create app file
print("\n[1/5] Creating Streamlit app file...")

app_code = '''"""
Breast Cancer Detection - Streamlit Web Application (FIXED)
Properly preprocesses images to match training pipeline
"""

import streamlit as st
import os
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF1493;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .benign-box {
        background: linear-gradient(135deg, #90EE90 0%, #32CD32 100%);
        border: 3px solid #228B22;
    }
    .malignant-box {
        background: linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%);
        border: 3px solid #DC143C;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_trained_model():
    """Load model with caching"""
    try:
        if os.path.exists('models/best_model.h5'):
            return load_model('models/best_model.h5'), 'best_model.h5'
    except Exception as e:
        st.error(f"Error: {e}")
    return None, None

def preprocess_image(img):
    """CRITICAL: Must match training preprocessing exactly"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Light denoising
    img = cv2.fastNlMeansDenoising(img, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    return img

def predict_image(model, img):
    """Make prediction on preprocessed image"""
    img_gray = preprocess_image(img)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)
    
    prediction = model.predict(img_rgb, verbose=0)
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class] * 100
    
    class_labels = {0: 'Benign', 1: 'Malignant'}
    label = class_labels[pred_class]
    
    return {
        'prediction': label,
        'confidence': float(confidence),
        'probabilities': {
            'benign': float(prediction[0][0] * 100),
            'malignant': float(prediction[0][1] * 100)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, img_gray

# Header
st.markdown('<p class="main-header">🎗️ Breast Cancer Detection System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Mammogram Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📋 Navigation")
    page = st.radio("", ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            model, model_name = load_trained_model()
            st.session_state.model = model
            st.session_state.model_name = model_name
    
    if st.session_state.model:
        st.success("✅ Model Ready")
        try:
            with open('models/model_metadata.json', 'r') as f:
                meta = json.load(f)
            st.metric("Test Accuracy", f"{meta.get('test_accuracy', 0)*100:.2f}%")
        except:
            pass
    else:
        st.error("❌ Model Not Found")

# Pages
if page == "🏠 Home":
    st.subheader("🎯 About This System")
    
    st.info("""
    This AI system analyzes mammogram images to detect potential breast cancer.
    It uses deep learning (ResNet50) trained on the CBIS-DDSM dataset.
    
    **Important:** This is for research/educational purposes only. 
    Always consult medical professionals for diagnosis.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Features:**")
        st.markdown("- Binary classification (Benign/Malignant)")
        st.markdown("- Confidence scores")
        st.markdown("- Prediction history")
        st.markdown("- Visual analysis")
    
    with col2:
        st.markdown("**⚠️ Limitations:**")
        st.markdown("- Not a medical device")
        st.markdown("- Research tool only")
        st.markdown("- Requires quality images")
        st.markdown("- Not 100% accurate")

elif page == "🔬 Prediction":
    st.header("🔬 Upload Mammogram for Analysis")
    
    if not st.session_state.model:
        st.error("❌ Model not loaded!")
        st.stop()
    
    uploaded = st.file_uploader(
        "Choose a mammogram image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a mammogram image (PNG/JPG)"
    )
    
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader("🔧 Preprocessing")
            st.info("""
            **Steps:**
            1. Convert to grayscale
            2. Resize to 224x224
            3. CLAHE enhancement
            4. Light denoising
            5. Normalization
            """)
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result, img_processed = predict_image(st.session_state.model, img)
                st.session_state.history.append(result)
                
                st.markdown("---")
                st.subheader("🖼️ Preprocessed Image")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_processed, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                st.markdown("---")
                
                if result['prediction'] == 'Benign':
                    st.markdown(f"""
                    <div class="prediction-box benign-box">
                        <h1>✅ BENIGN</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("The analysis suggests benign characteristics.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box malignant-box">
                        <h1>⚠️ MALIGNANT</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ This requires medical attention!")
                
                st.subheader("📊 Probabilities")
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=['Benign', 'Malignant'],
                        x=[result['probabilities']['benign'], 
                           result['probabilities']['malignant']],
                        orientation='h',
                        marker=dict(color=['green', 'red']),
                        text=[f"{result['probabilities']['benign']:.1f}%",
                              f"{result['probabilities']['malignant']:.1f}%"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(xaxis_title="Probability (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Benign", f"{result['probabilities']['benign']:.2f}%")
                with col2:
                    st.metric("Malignant", f"{result['probabilities']['malignant']:.2f}%")

elif page == "📊 Dashboard":
    st.header("📊 Prediction History")
    
    if not st.session_state.history:
        st.info("No predictions yet. Analyze images in the Prediction page.")
        st.stop()
    
    df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", len(df))
    with col2:
        benign = len(df[df['prediction'] == 'Benign'])
        st.metric("Benign", benign)
    with col3:
        malignant = len(df[df['prediction'] == 'Malignant'])
        st.metric("Malignant", malignant)
    
    st.dataframe(df, use_container_width=True)

elif page == "ℹ️ About":
    st.header("ℹ️ About")
    
    st.markdown("""
    ### 🎗️ Breast Cancer Detection System
    
    **Technology:** ResNet50 deep learning model
    **Dataset:** CBIS-DDSM
    **Purpose:** Research and education
    
    **⚠️ Disclaimer:**
    This is NOT a medical device. For research only.
    Always consult healthcare professionals.
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "© 2025 Breast Cancer Detection | Research & Educational Use Only</p>",
    unsafe_allow_html=True
)
'''

with open('5_streamlit_app.py', 'w') as f:
    f.write(app_code)

print("✅ Created: 5_streamlit_app.py")

# Step 2: Check dependencies
print("\n[2/5] Checking dependencies...")

required = ['streamlit', 'plotly', 'opencv-python', 'tensorflow', 'pillow']
missing = []

for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} (missing)")
        missing.append(pkg)

if missing:
    print(f"\n   Installing missing packages: {', '.join(missing)}")
    os.system(f"pip install -q {' '.join(missing)}")
    print("   ✅ Installation complete")

# Step 3: Check model directory
print("\n[3/5] Checking model directory...")

if not os.path.exists('models'):
    print("   ⚠️  'models' directory not found!")
    print("   Creating it now...")
    os.makedirs('models', exist_ok=True)
    print("   ✅ Created 'models' directory")
    print("\n   ⚠️  NOTE: You need to train a model first!")
    print("   The app will show 'Model Not Found' until you do.")
else:
    print("   ✅ 'models' directory exists")
    
    if os.path.exists('models/best_model.h5'):
        print("   ✅ Found: best_model.h5")
        size_mb = os.path.getsize('models/best_model.h5') / (1024*1024)
        print(f"   📊 Size: {size_mb:.1f} MB")
    else:
        print("   ⚠️  'best_model.h5' not found")
        print("   You need to train the model first!")

# Step 4: Test Streamlit import
print("\n[4/5] Testing Streamlit...")

try:
    import streamlit as st
    print(f"   ✅ Streamlit {st.__version__} ready")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 5: Summary
print("\n[5/5] Setup Summary")
print("=" * 80)

all_good = True

if os.path.exists('5_streamlit_app.py'):
    print("✅ App file: Ready")
else:
    print("❌ App file: Missing")
    all_good = False

if os.path.exists('models/best_model.h5'):
    print("✅ Model: Ready")
else:
    print("⚠️  Model: Not found (train it first!)")
    # Don't fail - app can still run

if not missing:
    print("✅ Dependencies: All installed")
else:
    print("❌ Dependencies: Some missing")
    all_good = False

print("=" * 80)

if all_good:
    print("\n🎉 SETUP COMPLETE!")
    print("\n📝 Next steps:")
    print("   1. Run the launcher script to start Streamlit")
    print("   2. Use LocalTunnel to access the app")
    print("   3. Upload mammogram images for analysis")
else:
    print("\n⚠️  SETUP INCOMPLETE")
    print("   Fix the issues above before launching")

print("\n" + "=" * 80)