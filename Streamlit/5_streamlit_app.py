"""
Breast Cancer Detection - Streamlit Web Application
Interactive dashboard for mammogram analysis
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
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
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
        if os.path.exists('models/best_model.keras'):
            model = load_model('models/best_model.keras')
            return model, 'best_model.keras'
        else:
            st.error("Model file not found! Train the model first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(img):
    """
    Preprocess image to match training pipeline
    CRITICAL: Must match training preprocessing exactly
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Light denoising
    img = cv2.fastNlMeansDenoising(img, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Normalize to 0-255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    return img

def predict_image(model, img):
    """Make prediction on preprocessed image"""
    # Preprocess
    img_gray = preprocess_image(img)
    
    # Convert to RGB (3 channels) for model
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1]
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_rgb = np.expand_dims(img_rgb, axis=0)
    
    # Predict
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
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Mammogram Analysis Using Deep Learning</p>', unsafe_allow_html=True)

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
            st.metric("AUC Score", f"{meta.get('test_auc', 0):.4f}")
        except:
            pass
    else:
        st.error("❌ Model Not Found")
        st.stop()

# Pages
if page == "🏠 Home":
    st.subheader("🎯 About This System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        This AI-powered system analyzes mammogram images to detect potential breast cancer.
        It uses **EfficientNetB3** deep learning architecture trained on the CBIS-DDSM dataset.
        
        **⚠️ Important:** This is for research and educational purposes only.
        Always consult qualified medical professionals for diagnosis and treatment.
        """)
    
    with col2:
        try:
            with open('models/model_metadata.json', 'r') as f:
                meta = json.load(f)
            st.markdown("### 📊 Model Stats")
            st.metric("Accuracy", f"{meta.get('test_accuracy', 0)*100:.1f}%")
            st.metric("Precision", f"{meta.get('test_precision', 0):.3f}")
            st.metric("Recall", f"{meta.get('test_recall', 0):.3f}")
        except:
            pass
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Features:**")
        st.markdown("- Binary classification (Benign/Malignant)")
        st.markdown("- Confidence scores with probabilities")
        st.markdown("- Prediction history tracking")
        st.markdown("- Visual preprocessing display")
        st.markdown("- Interactive dashboard")
    
    with col2:
        st.markdown("**⚠️ Limitations:**")
        st.markdown("- Not a medical device")
        st.markdown("- Research/educational tool only")
        st.markdown("- Requires quality mammogram images")
        st.markdown("- Cannot replace professional diagnosis")

elif page == "🔬 Prediction":
    st.header("🔬 Upload Mammogram for Analysis")
    
    uploaded = st.file_uploader(
        "Choose a mammogram image (PNG/JPG/JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a mammogram image for analysis"
    )
    
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader("🔧 Preprocessing Steps")
            st.info("""
            **Pipeline:**
            1. Grayscale conversion
            2. Resize to 224×224
            3. CLAHE enhancement
            4. Denoising
            5. Normalization
            """)
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing mammogram..."):
                result, img_processed = predict_image(st.session_state.model, img)
                st.session_state.history.append(result)
                
                # Show preprocessed image
                st.markdown("---")
                st.subheader("🖼️ Preprocessed Image")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img_processed, cmap='gray')
                    ax.axis('off')
                    ax.set_title('Enhanced Image', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                
                st.markdown("---")
                
                # Prediction result
                if result['prediction'] == 'Benign':
                    st.markdown(f"""
                    <div class="prediction-box benign-box">
                        <h1>✅ BENIGN</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ The analysis suggests benign characteristics.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box malignant-box">
                        <h1>⚠️ MALIGNANT</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ This requires immediate medical attention!")
                
                # Probability visualization
                st.subheader("📊 Class Probabilities")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Benign", f"{result['probabilities']['benign']:.2f}%")
                with col2:
                    st.metric("Malignant", f"{result['probabilities']['malignant']:.2f}%")
                
                # Bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        y=['Benign', 'Malignant'],
                        x=[result['probabilities']['benign'],
                           result['probabilities']['malignant']],
                        orientation='h',
                        marker=dict(
                            color=['#32CD32', '#FF1493'],
                            line=dict(color='#000', width=1)
                        ),
                        text=[f"{result['probabilities']['benign']:.1f}%",
                              f"{result['probabilities']['malignant']:.1f}%"],
                        textposition='auto',
                        textfont=dict(size=14, color='white')
                    )
                ])
                fig.update_layout(
                    xaxis_title="Probability (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Medical disclaimer
                st.warning("""
                **⚠️ Medical Disclaimer:** This analysis is generated by AI and should NOT be used 
                for medical diagnosis. Please consult with qualified healthcare professionals 
                for accurate diagnosis and treatment.
                """)

elif page == "📊 Dashboard":
    st.header("📊 Prediction History & Analytics")
    
    if not st.session_state.history:
        st.info("No predictions yet. Analyze images in the Prediction page.")
        st.stop()
    
    # Summary metrics
    df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        benign = len(df[df['prediction'] == 'Benign'])
        st.metric("Benign Cases", benign)
    with col3:
        malignant = len(df[df['prediction'] == 'Malignant'])
        st.metric("Malignant Cases", malignant)
    with col4:
        avg_conf = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    st.markdown("---")
    
    # Distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Classification Distribution")
        fig = go.Figure(data=[
            go.Pie(
                labels=['Benign', 'Malignant'],
                values=[benign, malignant],
                marker=dict(colors=['#32CD32', '#FF1493']),
                textinfo='label+percent',
                textfont=dict(size=14)
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Confidence Distribution")
        fig = go.Figure(data=[
            go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                marker=dict(color='#4CAF50', line=dict(color='#000', width=1))
            )
        ])
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Detailed History")
    st.dataframe(df, use_container_width=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

elif page == "ℹ️ About":
    st.header("ℹ️ About the System")
    
    st.markdown("""
    ### 🎗️ Breast Cancer Detection System
    
    **Purpose:** Research and educational tool for breast cancer detection using AI
    
    **Technology Stack:**
    - **Architecture:** EfficientNetB3 (Deep Learning)
    - **Framework:** TensorFlow/Keras
    - **Dataset:** CBIS-DDSM (Curated Breast Imaging Subset)
    - **Interface:** Streamlit
    
    **How It Works:**
    1. User uploads a mammogram image
    2. Image is preprocessed (CLAHE enhancement, denoising)
    3. Model analyzes the image
    4. System provides classification with confidence scores
    
    **⚠️ Important Disclaimers:**
    - This is NOT a medical device
    - For research and educational purposes ONLY
    - Cannot replace professional medical diagnosis
    - Always consult qualified healthcare professionals
    
    **Dataset Information:**
    - CBIS-DDSM: Curated Breast Imaging Subset of DDSM
    - Contains mammography images with benign and malignant cases
    - Images are preprocessed and enhanced for better analysis
    
    **Contact & Support:**
    For questions or feedback about this system, please consult the documentation
    or reach out through appropriate channels.
    """)
    
    try:
        with open('models/model_metadata.json', 'r') as f:
            meta = json.load(f)
        
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{meta.get('test_accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("Precision", f"{meta.get('test_precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{meta.get('test_recall', 0):.4f}")
        
        st.markdown(f"**Training Date:** {meta.get('training_date', 'N/A')}")
        st.markdown(f"**Total Epochs:** {meta.get('total_epochs', 'N/A')}")
    except:
        pass

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "© 2025 Breast Cancer Detection System | Research & Educational Use Only</p>",
    unsafe_allow_html=True
)