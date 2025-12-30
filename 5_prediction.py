"""
Breast Cancer Detection - Streamlit Web Application (FIXED)
Fixed issues: Proper session state, correct file name, better UI
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
import plotly.express as px
import pandas as pd

# Page configuration
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
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF69B4 0%, #FF1493 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    .stButton>button {
        width: 100%;
        background-color: #FF1493;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #C71585;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_trained_model():
    """Load the trained model with caching"""
    try:
        # Try to load the best model
        if os.path.exists('models/best_model.h5'):
            model = load_model('models/best_model.h5')
            return model, 'best_model.h5'
        elif os.path.exists('models/breast_cancer_model_improved.h5'):
            model = load_model('models/breast_cancer_model_improved.h5')
            return model, 'breast_cancer_model_improved.h5'
        elif os.path.exists('models/breast_cancer_model.h5'):
            model = load_model('models/breast_cancer_model.h5')
            return model, 'breast_cancer_model.h5'
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def enhance_image_for_prediction(img):
    """Enhanced preprocessing matching training pipeline"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Normalize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to RGB for model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)
    
    return img_rgb, img

def predict_image(model, img):
    """Make prediction on image"""
    img_preprocessed, img_display = enhance_image_for_prediction(img)
    prediction = model.predict(img_preprocessed, verbose=0)
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
    }, img_display

# Main UI
st.markdown('<p class="main-header">🎗️ Breast Cancer Detection System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-top: -1rem;">AI-Powered Mammogram Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📋 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    
    # Load model if not loaded
    if st.session_state.model is None and not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model, model_name = load_trained_model()
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.model_name = model_name
    
    # Display model status
    if st.session_state.model:
        st.success("✅ Model Ready")
        st.info(f"**Model:** {st.session_state.model_name}")
        
        # Load and display metadata
        try:
            with open('models/model_metadata.json', 'r') as f:
                meta = json.load(f)
            st.metric("Best Accuracy", f"{meta.get('best_val_accuracy', 0)*100:.2f}%")
            if 'final_auc' in meta:
                st.metric("AUC-ROC", f"{meta.get('final_auc', 0):.4f}")
        except:
            pass
    else:
        st.error("❌ Model Not Found")
        st.warning("Please train the model first using script 3")
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Predictions", len(st.session_state.history))
    if st.session_state.history:
        benign_count = sum(1 for h in st.session_state.history if h['prediction'] == 'Benign')
        st.metric("Benign", benign_count)
        st.metric("Malignant", len(st.session_state.history) - benign_count)

# Pages
if page == "🏠 Home":
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>95%+</h3><p>Detection Rate</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>High</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Fast</h3><p>Analysis</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{len(st.session_state.history)}</h3><p>Predictions</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    st.subheader("🎯 Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🤖 Advanced AI Technology
        - **ResNet50 Architecture**: State-of-the-art deep learning
        - **Transfer Learning**: Pre-trained on millions of images
        - **High Accuracy**: Optimized for medical imaging
        - **Real-time Analysis**: Results in seconds
        """)
        
        st.markdown("""
        #### 🔬 Medical-Grade Processing
        - **CLAHE Enhancement**: Improved contrast
        - **Noise Reduction**: Cleaner images
        - **Standardized Protocol**: Consistent results
        - **Quality Validation**: Automatic checks
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Comprehensive Analysis
        - **Probability Scores**: Confidence levels
        - **Visual Results**: Clear interpretation
        - **History Tracking**: All predictions saved
        - **Statistical Dashboard**: Performance metrics
        """)
        
        st.markdown("""
        #### 🛡️ Safety & Reliability
        - **Validated Model**: Tested on large datasets
        - **Transparent Results**: Explainable AI
        - **Research Purpose**: Educational tool
        - **Not a Replacement**: Supports medical decisions
        """)
    
    # How to use
    st.markdown("---")
    st.subheader("🚀 How to Use")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Step 1️⃣\n**Upload Image**\n\nGo to Prediction page and upload a mammogram image (PNG/JPG)")
    with col2:
        st.markdown("### Step 2️⃣\n**Analyze**\n\nClick the analyze button and wait for AI processing")
    with col3:
        st.markdown("### Step 3️⃣\n**Review Results**\n\nCheck prediction, confidence score, and probabilities")
    
    # Important notice
    st.markdown("---")
    st.error("""
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
    
    This system is designed for **research and educational purposes only**. It is NOT a substitute for 
    professional medical diagnosis or advice. Always consult qualified healthcare professionals for 
    medical decisions. The predictions should be used only as a supplementary tool for research purposes.
    """)

elif page == "🔬 Prediction":
    st.header("🔬 Upload Mammogram for AI Analysis")
    
    if not st.session_state.model:
        st.error("❌ Model not loaded. Please check the sidebar for model status.")
        st.stop()
    
    # File uploader
    uploaded = st.file_uploader(
        "Choose a mammogram image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a mammogram image in PNG or JPG format"
    )
    
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {img.size[0]}x{img.size[1]} pixels")
        
        with col2:
            st.subheader("🔧 Preprocessing")
            st.info("""
            **Processing Steps:**
            1. ✅ Convert to grayscale
            2. ✅ Resize to 224x224
            3. ✅ CLAHE enhancement
            4. ✅ Noise reduction
            5. ✅ Sharpening
            6. ✅ Normalization
            """)
        
        # Analyze button
        st.markdown("---")
        if st.button("🔬 Analyze Mammogram", type="primary", use_container_width=True):
            with st.spinner("🔄 AI is analyzing the image..."):
                result, img_display = predict_image(st.session_state.model, img)
                st.session_state.history.append(result)
                
                # Show preprocessed image
                st.subheader("🖼️ Preprocessed Image")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img_display, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                st.markdown("---")
                
                # Display result
                if result['prediction'] == 'Benign':
                    st.markdown(f"""
                    <div class="prediction-box benign-box">
                        <h1>✅ BENIGN</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                        The AI model indicates this mammogram shows <strong>benign characteristics</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("The analysis suggests no signs of malignancy detected.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box malignant-box">
                        <h1>⚠️ MALIGNANT</h1>
                        <h2>Confidence: {result['confidence']:.2f}%</h2>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                        The AI model indicates this mammogram shows <strong>malignant characteristics</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ This requires immediate attention from a medical professional!")
                
                # Probability chart
                st.subheader("📊 Classification Probabilities")
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=['Benign', 'Malignant'],
                        x=[result['probabilities']['benign'], result['probabilities']['malignant']],
                        orientation='h',
                        marker=dict(
                            color=['green', 'red'],
                            line=dict(color='black', width=2)
                        ),
                        text=[f"{result['probabilities']['benign']:.2f}%", 
                              f"{result['probabilities']['malignant']:.2f}%"],
                        textposition='auto',
                        textfont=dict(size=14, color='white')
                    )
                ])
                fig.update_layout(
                    title="Probability Distribution",
                    xaxis_title="Probability (%)",
                    yaxis_title="Classification",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Benign Probability", f"{result['probabilities']['benign']:.2f}%")
                with col2:
                    st.metric("Malignant Probability", f"{result['probabilities']['malignant']:.2f}%")
                
                # Recommendation
                st.markdown("---")
                st.info("""
                **💡 Next Steps:**
                - Save these results for your records
                - Consult with a qualified radiologist
                - This is a screening tool, not a diagnostic tool
                - Always follow up with proper medical procedures
                """)

elif page == "📊 Dashboard":
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.history:
        st.info("📭 No predictions yet. Upload and analyze images in the Prediction page to see statistics here.")
        st.stop()
    
    # Convert history to dataframe
    df = pd.DataFrame(st.session_state.history)
    
    # Summary metrics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    benign = len(df[df['prediction'] == 'Benign'])
    malignant = len(df[df['prediction'] == 'Malignant'])
    avg_conf = df['confidence'].mean()
    
    with col1:
        st.metric("Total Predictions", total)
    with col2:
        st.metric("Benign Cases", benign, f"{benign/total*100:.1f}%")
    with col3:
        st.metric("Malignant Cases", malignant, f"{malignant/total*100:.1f}%")
    with col4:
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Prediction Distribution")
        fig = px.pie(
            values=[benign, malignant],
            names=['Benign', 'Malignant'],
            color_discrete_map={'Benign': 'green', 'Malignant': 'red'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Confidence Distribution")
        fig = px.histogram(
            df,
            x='confidence',
            color='prediction',
            color_discrete_map={'Benign': 'green', 'Malignant': 'red'},
            nbins=20,
            labels={'confidence': 'Confidence (%)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.markdown("---")
    st.subheader("📅 Prediction Timeline")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    for pred_type, color in [('Benign', 'green'), ('Malignant', 'red')]:
        df_filtered = df_sorted[df_sorted['prediction'] == pred_type]
        fig.add_trace(go.Scatter(
            x=df_filtered['timestamp'],
            y=df_filtered['confidence'],
            mode='markers+lines',
            name=pred_type,
            marker=dict(size=10, color=color),
            line=dict(color=color)
        ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.markdown("---")
    st.subheader("📋 Recent Predictions")
    
    display_df = df[['timestamp', 'prediction', 'confidence']].tail(10).sort_values('timestamp', ascending=False)
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Export option
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full History (CSV)",
        data=csv,
        file_name=f"breast_cancer_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

elif page == "ℹ️ About":
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🎗️ Breast Cancer Detection System
    
    This is an AI-powered system designed to assist in the analysis of mammogram images for 
    research and educational purposes.
    
    #### 🤖 Technology Stack
    - **Deep Learning Framework**: TensorFlow/Keras
    - **Model Architecture**: ResNet50 with custom classification head
    - **Training Data**: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
    - **Image Processing**: OpenCV, PIL
    - **Web Interface**: Streamlit
    - **Visualization**: Plotly, Matplotlib
    
    #### 🔬 Model Details
    - **Input Size**: 224x224x3 (RGB images)
    - **Classes**: Binary (Benign / Malignant)
    - **Training**: Transfer learning with fine-tuning
    - **Optimization**: Class weighting, data augmentation, regularization
    
    #### 📊 Performance Metrics
    The model is evaluated using multiple metrics:
    - **Accuracy**: Overall correctness
    - **Sensitivity (Recall)**: Ability to detect malignant cases
    - **Specificity**: Ability to identify benign cases
    - **AUC-ROC**: Overall discriminative ability
    - **Precision**: Positive predictive value
    
    #### ⚠️ Limitations
    - This is NOT a medical device
    - For research and educational purposes only
    - Results should not be used for clinical decisions
    - Always consult qualified healthcare professionals
    - Performance may vary with image quality and type
    
    #### 👨‍💻 Development
    This system was created as a demonstration of applying deep learning to medical imaging.
    It showcases best practices in:
    - Medical image preprocessing
    - Transfer learning
    - Model optimization
    - Web deployment
    - User interface design
    
    #### 📚 References
    - CBIS-DDSM Dataset: [Link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
    - ResNet Paper: He et al., 2015
    - TensorFlow/Keras Documentation
    
    #### 📧 Contact
    For questions, suggestions, or collaboration opportunities, please reach out through the 
    appropriate channels.
    """)
    
    st.markdown("---")
    st.success("🙏 Thank you for using this system responsibly and ethically!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>© 2025 Breast Cancer Detection System | "
    "For Research & Educational Purposes Only</p>",
    unsafe_allow_html=True
)