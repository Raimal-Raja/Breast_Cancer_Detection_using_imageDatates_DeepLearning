"""
Breast Cancer Detection - Prediction Module
Handles image uploads, predictions, and self-learning queue
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class BreastCancerPredictor:
    def __init__(self, model_path='models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_labels = {0: 'Benign', 1: 'Malignant'}
        self.prediction_history = []
        
    def load_model(self):
        """Load trained model"""
        print("\n[LOADING] Loading trained model...")
        try:
            self.model = load_model(self.model_path)
            print(f"  ✓ Model loaded successfully from: {self.model_path}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            return False
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Preprocess image for prediction"""
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Try with PIL
            img = Image.open(img_path).convert('L')
            img = np.array(img)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Denoise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Convert to RGB (model expects 3 channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        img_rgb = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_rgb = np.expand_dims(img_rgb, axis=0)
        
        return img_rgb, img
    
    def predict(self, img_path, show_visualization=True):
        """Make prediction on single image"""
        print(f"\n[PREDICTION] Analyzing image: {os.path.basename(img_path)}")
        
        if self.model is None:
            print("  ✗ Model not loaded. Please load model first.")
            return None
        
        try:
            # Preprocess image
            img_preprocessed, img_display = self.preprocess_image(img_path)
            
            # Make prediction
            prediction = self.model.predict(img_preprocessed, verbose=0)
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class] * 100
            
            # Get label
            label = self.class_labels[pred_class]
            
            # Store prediction
            pred_result = {
                'image_path': img_path,
                'prediction': label,
                'confidence': float(confidence),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'probabilities': {
                    'benign': float(prediction[0][0] * 100),
                    'malignant': float(prediction[0][1] * 100)
                }
            }
            
            self.prediction_history.append(pred_result)
            
            # Print results
            print("\n  " + "=" * 50)
            print(f"  PREDICTION: {label}")
            print(f"  CONFIDENCE: {confidence:.2f}%")
            print("  " + "=" * 50)
            print(f"  Benign probability:    {pred_result['probabilities']['benign']:.2f}%")
            print(f"  Malignant probability: {pred_result['probabilities']['malignant']:.2f}%")
            print("  " + "=" * 50)
            
            # Save to retraining queue if malignant
            if label == 'Malignant' and confidence > 70:
                self._add_to_retraining_queue(img_path, pred_result)
            
            # Visualize
            if show_visualization:
                self.visualize_prediction(img_display, pred_result)
            
            return pred_result
            
        except Exception as e:
            print(f"  ✗ Error during prediction: {e}")
            return None
    
    def _add_to_retraining_queue(self, img_path, pred_result):
        """Add positive case to retraining queue"""
        print("\n  [SELF-LEARNING] Adding to retraining queue...")
        
        # Create queue directory
        queue_dir = 'models/retraining_queue'
        os.makedirs(queue_dir, exist_ok=True)
        
        # Copy image to queue
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        queue_filename = f"malignant_{timestamp}_{os.path.basename(img_path)}"
        queue_path = os.path.join(queue_dir, queue_filename)
        
        import shutil
        shutil.copy2(img_path, queue_path)
        
        # Save metadata
        metadata_path = os.path.join(queue_dir, f"{queue_filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(pred_result, f, indent=4)
        
        print(f"  ✓ Image added to retraining queue: {queue_filename}")
        
        # Check queue size
        queue_size = len([f for f in os.listdir(queue_dir) if f.endswith('.png') or f.endswith('.jpg')])
        print(f"  ✓ Current queue size: {queue_size} images")
        
        if queue_size >= 100:
            print("  ⚠ Queue threshold reached (100+ images). Consider retraining model.")
    
    def visualize_prediction(self, img, pred_result):
        """Visualize prediction result"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Display prediction
        labels = list(pred_result['probabilities'].keys())
        probs = list(pred_result['probabilities'].values())
        colors = ['green' if pred_result['prediction'] == 'Benign' else 'red', 
                 'red' if pred_result['prediction'] == 'Malignant' else 'green']
        
        axes[1].barh(labels, probs, color=colors, alpha=0.7)
        axes[1].set_xlabel('Probability (%)', fontsize=12)
        axes[1].set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 100])
        
        # Add values on bars
        for i, (label, prob) in enumerate(zip(labels, probs)):
            axes[1].text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)
        
        # Add prediction text
        fig.text(0.5, 0.02, 
                f"PREDICTION: {pred_result['prediction']} (Confidence: {pred_result['confidence']:.2f}%)",
                ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save
        result_filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f'results/{result_filename}', dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Visualization saved to 'results/{result_filename}'")
        
        plt.show()
    
    def batch_predict(self, image_folder, save_report=True):
        """Predict on multiple images"""
        print(f"\n[BATCH PREDICTION] Processing images from: {image_folder}")
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("  ✗ No images found in folder")
            return []
        
        print(f"  Found {len(image_files)} images")
        
        # Process each image
        results = []
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            result = self.predict(img_path, show_visualization=False)
            if result:
                results.append(result)
        
        # Generate summary
        if results and save_report:
            self._generate_batch_report(results)
        
        return results
    
    def _generate_batch_report(self, results):
        """Generate report for batch predictions"""
        print("\n[REPORT] Generating batch prediction report...")
        
        # Calculate statistics
        total = len(results)
        benign_count = sum(1 for r in results if r['prediction'] == 'Benign')
        malignant_count = sum(1 for r in results if r['prediction'] == 'Malignant')
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Create report
        report = {
            'total_images': total,
            'benign_count': benign_count,
            'malignant_count': malignant_count,
            'benign_percentage': (benign_count / total * 100) if total > 0 else 0,
            'malignant_percentage': (malignant_count / total * 100) if total > 0 else 0,
            'average_confidence': float(avg_confidence),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': results
        }
        
        # Save report
        report_filename = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f'results/{report_filename}', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\n  " + "=" * 60)
        print("  BATCH PREDICTION SUMMARY")
        print("  " + "=" * 60)
        print(f"  Total images:      {total}")
        print(f"  Benign:            {benign_count} ({report['benign_percentage']:.1f}%)")
        print(f"  Malignant:         {malignant_count} ({report['malignant_percentage']:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.2f}%")
        print("  " + "=" * 60)
        print(f"\n  ✓ Report saved to 'results/{report_filename}'")
    
    def check_retraining_queue(self):
        """Check status of retraining queue"""
        queue_dir = 'models/retraining_queue'
        
        if not os.path.exists(queue_dir):
            print("\n[QUEUE STATUS] Retraining queue is empty")
            return
        
        queue_images = [f for f in os.listdir(queue_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print("\n" + "=" * 60)
        print("RETRAINING QUEUE STATUS")
        print("=" * 60)
        print(f"Images in queue: {len(queue_images)}")
        print(f"Threshold: 100 images")
        print(f"Status: {'READY FOR RETRAINING' if len(queue_images) >= 100 else 'Collecting data...'}")
        print("=" * 60)

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER DETECTION - PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    predictor = BreastCancerPredictor(model_path='models/best_model.h5')
    
    # Load model
    if not predictor.load_model():
        print("\n✗ Failed to load model. Please train the model first.")
        exit(1)
    
    # Example: Predict on a single image
    print("\n[MODE] Single Image Prediction")
    print("Please specify the path to your test image")
    print("Example: 'data/user_uploads/test_image.png'")
    
    # For demo purposes, let's check if user_uploads folder has images
    user_uploads = 'data/user_uploads'
    os.makedirs(user_uploads, exist_ok=True)
    
    test_images = [f for f in os.listdir(user_uploads) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if test_images:
        print(f"\nFound {len(test_images)} test images in user_uploads folder")
        
        # Predict on first image
        test_img_path = os.path.join(user_uploads, test_images[0])
        result = predictor.predict(test_img_path, show_visualization=True)
        
        # Batch predict if multiple images
        if len(test_images) > 1:
            print("\n[MODE] Batch Prediction on all uploaded images")
            results = predictor.batch_predict(user_uploads, save_report=True)
    else:
        print("\n✗ No images found in 'data/user_uploads/' folder")
        print("\nTo test predictions:")
        print("1. Upload mammogram images to 'data/user_uploads/' folder")
        print("2. Run this script again")
    
    # Check retraining queue status
    predictor.check_retraining_queue()
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run '6_streamlit_app.py' to launch web interface")