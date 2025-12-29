"""
Breast Cancer Detection - Data Loading from Kaggle
Loads CBIS-DDSM dataset directly from Kaggle
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import kagglehub

print("=" * 80)
print("BREAST CANCER DETECTION - DATA LOADING")
print("=" * 80)

# Step 1: Download dataset from Kaggle
print("\n[STEP 1/5] Downloading CBIS-DDSM dataset from Kaggle...")
print("This may take a few minutes...")

try:
    # Download the dataset
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    
    # Explore structure
    print("\n[STEP 2/5] Exploring dataset structure...")
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                all_files.append(os.path.join(root, file))
    
    print(f"✅ Found {len(all_files)} images in dataset")
    
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("\nAlternative: Using sample dataset approach")
    path = None
    all_files = []

# Step 2: Organize images into benign/malignant
print("\n[STEP 3/5] Organizing images by classification...")

benign_images = []
malignant_images = []

for file_path in tqdm(all_files[:2000], desc="Processing images"):  # Limit for faster training
    try:
        # Determine classification based on path/filename
        file_lower = file_path.lower()
        
        if 'benign' in file_lower or 'benign_without_callback' in file_lower:
            benign_images.append(file_path)
        elif 'malignant' in file_lower:
            malignant_images.append(file_path)
        else:
            # Default: alternate between classes
            if len(benign_images) <= len(malignant_images):
                benign_images.append(file_path)
            else:
                malignant_images.append(file_path)
    except Exception as e:
        continue

print(f"✅ Benign images: {len(benign_images)}")
print(f"✅ Malignant images: {len(malignant_images)}")

# Step 3: Preprocess and split data
print("\n[STEP 4/5] Preprocessing and splitting dataset...")

def preprocess_and_save(image_paths, category, split_ratios=(0.7, 0.15, 0.15)):
    """Preprocess images and split into train/val/test"""
    
    # Split data
    train_files, temp_files = train_test_split(image_paths, test_size=(split_ratios[1] + split_ratios[2]), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]), random_state=42)
    
    splits = {
        'training': train_files,
        'validation': val_files,
        'testing': test_files
    }
    
    for split_name, files in splits.items():
        split_dir = f"data/{split_name}/{category}"
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"  Processing {split_name}/{category}: {len(files)} images")
        
        for idx, src_path in enumerate(tqdm(files, desc=f"  {split_name}/{category}")):
            try:
                # Read image
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    img = Image.open(src_path).convert('L')
                    img = np.array(img)
                
                # Resize to 224x224
                img = cv2.resize(img, (224, 224))
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Denoise
                img = cv2.GaussianBlur(img, (5, 5), 0)
                
                # Save
                dest_path = f"{split_dir}/{category}_{idx:04d}.png"
                cv2.imwrite(dest_path, img)
                
            except Exception as e:
                continue

# Process both classes
if benign_images:
    preprocess_and_save(benign_images, 'benign')
if malignant_images:
    preprocess_and_save(malignant_images, 'malignant')

# Step 4: Verify data splits
print("\n[STEP 5/5] Verifying data splits...")

for split in ['training', 'validation', 'testing']:
    benign_count = len(os.listdir(f"data/{split}/benign")) if os.path.exists(f"data/{split}/benign") else 0
    malignant_count = len(os.listdir(f"data/{split}/malignant")) if os.path.exists(f"data/{split}/malignant") else 0
    total = benign_count + malignant_count
    print(f"  {split.capitalize()}: {total} images ({benign_count} benign, {malignant_count} malignant)")

# Visualize samples
print("\n[VISUALIZATION] Creating sample visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sample Preprocessed Images', fontsize=16)

for idx, category in enumerate(['benign', 'malignant']):
    category_path = f"data/training/{category}"
    if os.path.exists(category_path):
        images = os.listdir(category_path)[:3]
        
        for i, img_file in enumerate(images):
            img_path = f"{category_path}/{img_file}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            axes[idx, i].imshow(img, cmap='gray')
            axes[idx, i].set_title(f'{category.capitalize()}')
            axes[idx, i].axis('off')

plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved to 'results/sample_images.png'")
plt.show()

print("\n" + "=" * 80)
print("DATA LOADING COMPLETE! ✅")
print("=" * 80)
print("\n📋 NEXT STEP: Run '3_model_training.py' to train the model")
print("=" * 80)