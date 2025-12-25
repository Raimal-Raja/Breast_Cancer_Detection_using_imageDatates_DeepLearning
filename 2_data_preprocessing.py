"""
Breast Cancer Detection - Data Preprocessing Module
Loads CBIS-DDSM dataset and prepares data for training
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil
from tqdm import tqdm
import json

class DataPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.stats = {
            'total_images': 0,
            'benign': 0,
            'malignant': 0,
            'normal': 0
        }
    
    def load_dataset_from_kaggle(self):
        """Load CBIS-DDSM dataset from Kaggle"""
        print("\n[STEP 1] Loading CBIS-DDSM dataset from Kaggle...")
        
        try:
            # Download dataset
            path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
            print(f"  ✓ Dataset downloaded to: {path}")
            
            # Explore dataset structure
            print("\n  Dataset structure:")
            for root, dirs, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                if level < 2:  # Limit depth
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:3]:  # Show first 3 files
                        print(f'{subindent}{file}')
                    if len(files) > 3:
                        print(f'{subindent}... and {len(files)-3} more files')
            
            return path
            
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            print("\n  Trying alternative approach...")
            # Alternative: Use sample dataset structure
            return self._create_sample_structure()
    
    def _create_sample_structure(self):
        """Create sample structure if Kaggle download fails"""
        print("  Creating sample dataset structure...")
        sample_path = "data/raw/sample"
        os.makedirs(f"{sample_path}/benign", exist_ok=True)
        os.makedirs(f"{sample_path}/malignant", exist_ok=True)
        print(f"  ✓ Sample structure created at: {sample_path}")
        print("\n  NOTE: Please manually add mammogram images to:")
        print(f"    - {sample_path}/benign/")
        print(f"    - {sample_path}/malignant/")
        return sample_path
    
    def organize_dataset(self, source_path):
        """Organize images into benign/malignant folders"""
        print("\n[STEP 2] Organizing dataset...")
        
        # Create organized structure
        organized_path = "data/raw/organized"
        categories = ['benign', 'malignant']
        
        for category in categories:
            os.makedirs(f"{organized_path}/{category}", exist_ok=True)
        
        # Find and organize images
        image_count = {'benign': 0, 'malignant': 0}
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    # Determine category from path or filename
                    file_path = os.path.join(root, file)
                    
                    if 'benign' in root.lower() or 'benign' in file.lower():
                        category = 'benign'
                    elif 'malignant' in root.lower() or 'malignant' in file.lower():
                        category = 'malignant'
                    else:
                        # Default categorization
                        category = 'benign' if image_count['benign'] <= image_count['malignant'] else 'malignant'
                    
                    # Copy image
                    dest_path = f"{organized_path}/{category}/{category}_{image_count[category]:04d}.png"
                    try:
                        img = Image.open(file_path).convert('L')  # Convert to grayscale
                        img.save(dest_path)
                        image_count[category] += 1
                    except Exception as e:
                        print(f"  Warning: Could not process {file}: {e}")
        
        print(f"  ✓ Organized {image_count['benign']} benign images")
        print(f"  ✓ Organized {image_count['malignant']} malignant images")
        
        self.stats['benign'] = image_count['benign']
        self.stats['malignant'] = image_count['malignant']
        self.stats['total_images'] = sum(image_count.values())
        
        return organized_path
    
    def preprocess_image(self, image_path):
        """Apply preprocessing to single image"""
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Resize
        img = cv2.resize(img, self.target_size)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Denoise with Gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def augment_and_split_data(self, organized_path):
        """Apply data augmentation and split into train/val/test"""
        print("\n[STEP 3] Augmenting and splitting dataset...")
        
        # Data augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Process each category
        for category in ['benign', 'malignant']:
            print(f"\n  Processing {category} images...")
            category_path = f"{organized_path}/{category}"
            image_files = [f for f in os.listdir(category_path) if f.endswith('.png')]
            
            # Split files
            train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
            
            splits = {
                'training': train_files,
                'validation': val_files,
                'testing': test_files
            }
            
            for split_name, files in splits.items():
                split_path = f"data/{split_name}/{category}"
                os.makedirs(split_path, exist_ok=True)
                
                print(f"    Processing {split_name}: {len(files)} images")
                
                for file in tqdm(files, desc=f"    {split_name}"):
                    src_path = f"{category_path}/{file}"
                    
                    # Preprocess
                    img = self.preprocess_image(src_path)
                    
                    if img is not None:
                        # Save preprocessed image
                        dest_path = f"{split_path}/{file}"
                        cv2.imwrite(dest_path, (img * 255).astype(np.uint8))
                        
                        # Apply augmentation only for training set
                        if split_name == 'training':
                            img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                            img_rgb = np.expand_dims(img_rgb, axis=0)
                            
                            # Generate augmented images
                            aug_count = 0
                            for batch in datagen.flow(img_rgb, batch_size=1):
                                aug_img = batch[0].astype(np.uint8)
                                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)
                                
                                aug_path = f"{split_path}/aug_{aug_count}_{file}"
                                cv2.imwrite(aug_path, aug_img)
                                
                                aug_count += 1
                                if aug_count >= 5:  # Generate 5 augmented versions
                                    break
        
        print("\n  ✓ Data augmentation and splitting complete!")
        self._print_split_stats()
    
    def _print_split_stats(self):
        """Print statistics about the data splits"""
        print("\n  Dataset statistics:")
        for split in ['training', 'validation', 'testing']:
            benign_count = len(os.listdir(f"data/{split}/benign")) if os.path.exists(f"data/{split}/benign") else 0
            malignant_count = len(os.listdir(f"data/{split}/malignant")) if os.path.exists(f"data/{split}/malignant") else 0
            print(f"    {split.capitalize()}: {benign_count + malignant_count} images ({benign_count} benign, {malignant_count} malignant)")
    
    def visualize_samples(self):
        """Visualize sample preprocessed images"""
        print("\n[STEP 4] Visualizing sample images...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Sample Preprocessed Images', fontsize=16)
        
        categories = ['benign', 'malignant']
        
        for idx, category in enumerate(categories):
            category_path = f"data/training/{category}"
            if os.path.exists(category_path):
                images = os.listdir(category_path)[:3]
                
                for i, img_file in enumerate(images):
                    img_path = f"{category_path}/{img_file}"
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    axes[idx, i].imshow(img, cmap='gray')
                    axes[idx, i].set_title(f'{category.capitalize()} - {img_file}')
                    axes[idx, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
        print("  ✓ Sample visualization saved to 'results/sample_images.png'")
        plt.show()
    
    def save_preprocessing_metadata(self):
        """Save preprocessing statistics"""
        metadata = {
            'target_size': self.target_size,
            'stats': self.stats,
            'preprocessing_steps': [
                'Resize to 224x224',
                'CLAHE enhancement',
                'Gaussian blur denoising',
                'Normalization to [0, 1]',
                'Data augmentation (training only)'
            ]
        }
        
        with open('models/preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("\n  ✓ Preprocessing metadata saved")

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(target_size=(224, 224))
    
    # Step 1: Load dataset
    dataset_path = preprocessor.load_dataset_from_kaggle()
    
    # Step 2: Organize dataset
    organized_path = preprocessor.organize_dataset(dataset_path)
    
    # Step 3: Augment and split
    preprocessor.augment_and_split_data(organized_path)
    
    # Step 4: Visualize samples
    preprocessor.visualize_samples()
    
    # Save metadata
    preprocessor.save_preprocessing_metadata()
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run '3_automl_training.py' to train the model")