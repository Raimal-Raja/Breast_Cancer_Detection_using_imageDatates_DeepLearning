"""
Breast Cancer Detection - Data Loading (FIXED - CSV Matching)
Properly matches CSV metadata with actual image files
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
print("BREAST CANCER DETECTION - DATA LOADING (FIXED)")
print("=" * 80)

# Step 1: Download dataset from Kaggle
print("\n[STEP 1/7] Downloading CBIS-DDSM dataset from Kaggle...")
print("⏳ This may take 5-10 minutes...")

try:
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Step 2: Load ALL CSV files and combine them
print("\n[STEP 2/7] Loading and combining metadata CSV files...")

csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.csv') and 'case_description' in file:
            csv_files.append(os.path.join(root, file))

print(f"Found {len(csv_files)} metadata CSV files")

# Load and combine all CSVs
all_metadata = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        if 'pathology' in df.columns:
            print(f"\n✅ Loading: {os.path.basename(csv_file)}")
            print(f"   Records: {len(df)}")
            print(f"   Pathology distribution:")
            for path_type, count in df['pathology'].value_counts().items():
                print(f"     - {path_type}: {count}")
            all_metadata.append(df)
    except Exception as e:
        print(f"⚠️  Error loading {csv_file}: {e}")

if not all_metadata:
    print("❌ No valid CSV metadata found!")
    exit(1)

# Combine all metadata
df_combined = pd.concat(all_metadata, ignore_index=True)
print(f"\n✅ Total metadata records: {len(df_combined)}")
print(f"Columns: {list(df_combined.columns)}")

# Step 3: Find all actual image files
print("\n[STEP 3/7] Scanning for image files...")

all_image_files = {}  # filename -> full_path mapping
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
            full_path = os.path.join(root, file)
            # Store both with and without extension
            base_name = os.path.splitext(file)[0]
            all_image_files[file] = full_path
            all_image_files[base_name] = full_path

print(f"✅ Found {len(all_image_files)} image file mappings")

# Step 4: Match CSV records with actual image files
print("\n[STEP 4/7] Matching CSV records with images...")

benign_images = []
malignant_images = []
matched_count = 0
unmatched_count = 0

for idx, row in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Matching"):
    try:
        # Get pathology
        pathology = str(row['pathology']).upper()
        
        # Determine class
        is_benign = 'BENIGN' in pathology
        is_malignant = 'MALIGNANT' in pathology
        
        if not (is_benign or is_malignant):
            continue
        
        # Try to find the image file
        # CSV has paths like: 'Mass-Training_P_00001_LEFT_CC/1.3.6.1...'
        image_path = None
        
        # Try different path columns
        path_columns = ['image file path', 'cropped image file path', 
                       'ROI mask file path']
        
        for col in path_columns:
            if col in row and pd.notna(row[col]):
                csv_path = str(row[col])
                
                # Extract filename from CSV path
                if '/' in csv_path:
                    parts = csv_path.split('/')
                    # Try each part as a potential filename
                    for part in parts:
                        if part in all_image_files:
                            image_path = all_image_files[part]
                            break
                        # Try without extension
                        base = os.path.splitext(part)[0]
                        if base in all_image_files:
                            image_path = all_image_files[base]
                            break
                    if image_path:
                        break
                else:
                    # Direct filename
                    if csv_path in all_image_files:
                        image_path = all_image_files[csv_path]
                        break
        
        # If still not found, try searching by partial path
        if not image_path:
            for col in path_columns:
                if col in row and pd.notna(row[col]):
                    csv_path = str(row[col])
                    # Search for any image file containing part of this path
                    for file_key, file_path in all_image_files.items():
                        if any(part in file_path for part in csv_path.split('/')):
                            image_path = file_path
                            break
                    if image_path:
                        break
        
        # Add to appropriate list
        if image_path and os.path.exists(image_path):
            matched_count += 1
            if is_malignant and not is_benign:
                malignant_images.append(image_path)
            elif is_benign and not is_malignant:
                benign_images.append(image_path)
        else:
            unmatched_count += 1
            
    except Exception as e:
        continue

print(f"\n✅ Matching complete:")
print(f"   Matched: {matched_count}")
print(f"   Unmatched: {unmatched_count}")
print(f"   Benign images: {len(benign_images)}")
print(f"   Malignant images: {len(malignant_images)}")

# If CSV matching failed, fall back to directory-based classification
if len(benign_images) == 0 and len(malignant_images) == 0:
    print("\n⚠️  CSV matching failed. Using directory-based classification...")
    
    benign_images = []
    malignant_images = []
    
    for file_path in tqdm(list(all_image_files.values()), desc="Classifying by directory"):
        path_lower = file_path.lower()
        
        # Check directory structure
        if 'benign' in path_lower and 'malignant' not in path_lower:
            benign_images.append(file_path)
        elif 'malignant' in path_lower or 'cancer' in path_lower:
            malignant_images.append(file_path)
    
    print(f"\n✅ Directory-based classification:")
    print(f"   Benign images: {len(benign_images)}")
    print(f"   Malignant images: {len(malignant_images)}")

# Check if we have data
if len(benign_images) == 0 or len(malignant_images) == 0:
    print("\n❌ ERROR: Unable to find images for one or both classes!")
    print("\nDEBUG: Let's check the directory structure...")
    
    # Show directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 3:  # Only show first 3 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show only first 3 files
                print(f'{subindent}{file}')
            if len(files) > 3:
                print(f'{subindent}... and {len(files)-3} more files')
    
    exit(1)

# Step 5: Balance dataset
print("\n[STEP 5/7] Balancing dataset...")

min_samples = min(len(benign_images), len(malignant_images))
max_per_class = min(min_samples, 2000)

if max_per_class < 100:
    print(f"⚠️  WARNING: Very few images ({max_per_class} per class)")
    print("   Model performance may be limited")

np.random.seed(42)
benign_images = np.random.choice(benign_images, max_per_class, replace=False).tolist()
malignant_images = np.random.choice(malignant_images, max_per_class, replace=False).tolist()

print(f"✅ Balanced to {max_per_class} images per class")
print(f"   Total dataset: {len(benign_images) + len(malignant_images)} images")

# Step 6: Preprocessing function
def preprocess_image(img_path):
    """Enhanced preprocessing with variation preservation"""
    try:
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
        
        if img is None or img.size == 0:
            return None
        
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
        
    except Exception as e:
        return None

def split_and_save_data(image_paths, label, split_ratios=(0.7, 0.15, 0.15)):
    """Split and save preprocessed images"""
    
    if len(image_paths) == 0:
        print(f"⚠️  No images to process for {label}")
        return {}
    
    # Split data
    train_imgs, temp_imgs = train_test_split(
        image_paths,
        test_size=(split_ratios[1] + split_ratios[2]),
        random_state=42,
        shuffle=True
    )
    
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        random_state=42,
        shuffle=True
    )
    
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    stats = {}
    
    for split_name, imgs in splits.items():
        save_dir = f"data/{split_name}/{label}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n  Processing {split_name}/{label}: {len(imgs)} images")
        
        successful = 0
        for idx, src_path in enumerate(tqdm(imgs, desc=f"  {split_name}/{label}")):
            processed_img = preprocess_image(src_path)
            
            if processed_img is not None:
                dest_path = f"{save_dir}/{label}_{idx:05d}.png"
                cv2.imwrite(dest_path, processed_img)
                successful += 1
        
        stats[split_name] = successful
        print(f"    ✅ Saved: {successful}/{len(imgs)}")
    
    return stats

# Step 7: Process both classes
print("\n[STEP 6/7] Preprocessing and saving images...")

print("\n📊 Processing BENIGN images:")
benign_stats = split_and_save_data(benign_images, 'benign')

print("\n📊 Processing MALIGNANT images:")
malignant_stats = split_and_save_data(malignant_images, 'malignant')

# Verify splits
print("\n[STEP 7/7] Verifying data splits...")
print("\n" + "=" * 80)
print("DATASET DISTRIBUTION")
print("=" * 80)

total_images = 0
for split in ['train', 'val', 'test']:
    benign_path = f"data/{split}/benign"
    malignant_path = f"data/{split}/malignant"
    
    benign_count = len(os.listdir(benign_path)) if os.path.exists(benign_path) else 0
    malignant_count = len(os.listdir(malignant_path)) if os.path.exists(malignant_path) else 0
    total = benign_count + malignant_count
    total_images += total
    
    balance_ratio = (benign_count / malignant_count * 100) if malignant_count > 0 else 0
    
    print(f"{split.upper():8} | Total: {total:4} | Benign: {benign_count:4} | Malignant: {malignant_count:4} | Ratio: {balance_ratio:.1f}%")

print("=" * 80)
print(f"TOTAL DATASET: {total_images} images")
print("=" * 80)

if total_images == 0:
    print("\n❌ ERROR: No images were processed!")
    exit(1)

# Visualize samples
print("\n[VISUALIZATION] Creating sample images...")

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Sample Preprocessed Images', fontsize=16, fontweight='bold')

for row in range(4):
    for col in range(4):
        if col < 2:
            img_dir = 'data/train/benign'
            label = 'Benign'
            color = 'green'
        else:
            img_dir = 'data/train/malignant'
            label = 'Malignant'
            color = 'red'
        
        if os.path.exists(img_dir):
            images = sorted(os.listdir(img_dir))
            if len(images) > row * 2 + (col % 2):
                img_file = images[row * 2 + (col % 2)]
                img_path = os.path.join(img_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f'{label} #{row * 2 + (col % 2) + 1}',
                                        fontsize=10, color=color, fontweight='bold')
                axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved")
plt.show()

# Save dataset info
train_benign = len(os.listdir('data/train/benign')) if os.path.exists('data/train/benign') else 0
train_malignant = len(os.listdir('data/train/malignant')) if os.path.exists('data/train/malignant') else 0

dataset_info = {
    'total_images': total_images,
    'train_images': train_benign + train_malignant,
    'val_images': len(os.listdir('data/val/benign')) + len(os.listdir('data/val/malignant')),
    'test_images': len(os.listdir('data/test/benign')) + len(os.listdir('data/test/malignant')),
    'benign_train': train_benign,
    'malignant_train': train_malignant,
    'preprocessing': 'Moderate CLAHE, Light denoising, Normalization'
}

import json
with open('data/dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=4)

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE! ✅")
print("=" * 80)
print("\n📋 NEXT STEP: Run '3_model_training.py' to train the model")
print("=" * 80)