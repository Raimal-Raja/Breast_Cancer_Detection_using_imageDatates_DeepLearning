# 🚀 Quick Start Guide - Breast Cancer Detection

## ⚡ Fast Track Setup (30 Minutes)

### Part 1: Train Model on Google Colab (20 min)

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com/)

2. **Create Notebook**: `File` → `New notebook`

3. **Enable GPU**: `Runtime` → `Change runtime type` → Select `GPU`

4. **Copy & Paste**: Copy the entire code from `breast_cancer_colab_training.ipynb`

5. **Run**: Press `Shift + Enter` and wait (~2-3 hours with GPU)

6. **Download Models**: After training, download from Google Drive:
   ```
   MyDrive/BreastCancerDetection/models/
   ├── breast_cancer_model.h5
   └── model_config.pkl
   ```

### Part 2: Run Flask App Locally (10 min)

**Windows:**
```cmd
# Create project folder
mkdir BreastCancerDetection\flask_app
cd BreastCancerDetection\flask_app

# Create subfolders
mkdir templates static models uploads
mkdir static\css static\js

# Copy all files (app.py, index.html, style.css, script.js, requirements.txt)
# Copy downloaded model files to models\ folder

# Install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run app
python app.py
```

**Mac/Linux:**
```bash
# Create project folder
mkdir -p BreastCancerDetection/flask_app
cd BreastCancerDetection/flask_app

# Create subfolders
mkdir -p templates static/{css,js} models uploads

# Copy all files (app.py, index.html, style.css, script.js, requirements.txt)
# Copy downloaded model files to models/ folder

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run app
python app.py
```

7. **Open Browser**: Navigate to `http://localhost:5000`

---

## 📂 Required Files Checklist

### Files to Create:
- [ ] `flask_app/app.py`
- [ ] `flask_app/templates/index.html`
- [ ] `flask_app/static/css/style.css`
- [ ] `flask_app/static/js/script.js`
- [ ] `flask_app/requirements.txt`

### Files to Download from Colab:
- [ ] `flask_app/models/breast_cancer_model.h5`
- [ ] `flask_app/models/model_config.pkl`

---

## ✅ Verification Steps

1. **Model Loaded**: Check console for "✓ Model loaded successfully!"
2. **Accuracy**: Should show 90%+ accuracy
3. **Web Interface**: Should load at localhost:5000
4. **Upload Test**: Try uploading a sample image

---

## 🐛 Common Quick Fixes

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow==2.13.0
```

**Problem**: `FileNotFoundError: breast_cancer_model.h5`
- Make sure model files are in `models/` folder
- Check file names exactly match

**Problem**: Port 5000 already in use
- Change port in app.py: `app.run(port=5001)`

**Problem**: Colab disconnects
- Keep browser tab active
- Or reduce dataset: `sample_fraction=0.3`

---

## 📊 Expected Results

**Training Output:**
```
Test Accuracy: 91.23%
Test AUC: 0.9654
Test Precision: 0.8945
Test Recall: 0.8723
```

**Flask Output:**
```
BREAST CANCER DETECTION - FLASK APPLICATION
Model Accuracy: 91.23%
Model AUC: 0.9654
Running on http://127.0.0.1:5000
```

---

## 🎯 Testing

1. Upload a histopathology image
2. Wait 2-3 seconds for prediction
3. See results:
   - **Prediction**: IDC Positive / No IDC
   - **Confidence**: Percentage confidence
   - **Probabilities**: Breakdown by class

---

## 💡 Tips

- **First Time**: Use `sample_fraction=0.3` in Colab for faster testing (~30 min training)
- **GPU**: Always use GPU in Colab for faster training
- **File Size**: Keep images under 16MB
- **Formats**: Use PNG, JPG, JPEG, BMP, or TIFF

---

## 📞 Need Help?

Refer to the complete `COMPLETE_SETUP_GUIDE.md` for detailed instructions and troubleshooting.

---

**That's it!** You now have a working breast cancer detection system! 🎉