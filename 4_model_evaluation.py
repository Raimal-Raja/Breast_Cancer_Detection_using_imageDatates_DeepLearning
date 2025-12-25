"""
Breast Cancer Detection - Model Evaluation Module
Evaluates model performance on test set
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, model_path='models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_labels = ['benign', 'malignant']
        
    def load_model(self):
        """Load trained model"""
        print("\n[STEP 1] Loading trained model...")
        try:
            self.model = load_model(self.model_path)
            print(f"  ✓ Model loaded from: {self.model_path}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            return False
    
    def prepare_test_data(self, batch_size=32):
        """Prepare test data generator"""
        print("\n[STEP 2] Preparing test data...")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            'data/testing',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"  ✓ Test samples: {test_generator.samples}")
        print(f"  ✓ Classes: {test_generator.class_indices}")
        
        return test_generator
    
    def evaluate_model(self, test_gen):
        """Evaluate model on test set"""
        print("\n[STEP 3] Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("\n  Model Performance Metrics:")
        print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  ✓ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn)  # Same as recall for positive class
        specificity = tn / (tn + fp)
        
        print(f"  ✓ Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"  ✓ Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        
        # Detailed classification report
        print("\n  Detailed Classification Report:")
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_labels,
            digits=4
        )
        print(report)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        print("\n[STEP 4] Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("  ✓ Confusion matrix saved to 'results/confusion_matrix.png'")
        plt.show()
    
    def plot_roc_curve(self, y_true, predictions):
        """Plot ROC curve"""
        print("\n[STEP 5] Generating ROC curve...")
        
        # Calculate ROC curve for malignant class (class 1)
        fpr, tpr, thresholds = roc_curve(y_true, predictions[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC curve saved (AUC: {roc_auc:.4f})")
        plt.show()
        
        return roc_auc
    
    def visualize_predictions(self, test_gen, num_samples=9):
        """Visualize sample predictions"""
        print("\n[STEP 6] Visualizing sample predictions...")
        
        # Get sample images
        test_gen.reset()
        batch_images, batch_labels = next(test_gen)
        predictions = self.model.predict(batch_images[:num_samples])
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        
        for idx in range(num_samples):
            row = idx // 3
            col = idx % 3
            
            # Get image and predictions
            img = batch_images[idx]
            true_label = self.class_labels[np.argmax(batch_labels[idx])]
            pred_label = self.class_labels[np.argmax(predictions[idx])]
            confidence = np.max(predictions[idx]) * 100
            
            # Plot
            axes[row, col].imshow(img)
            
            # Color code: green if correct, red if incorrect
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
            axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
        print("  ✓ Sample predictions saved to 'results/sample_predictions.png'")
        plt.show()
    
    def save_evaluation_results(self, results, roc_auc):
        """Save evaluation results to JSON"""
        print("\n[STEP 7] Saving evaluation results...")
        
        evaluation_data = {
            'test_metrics': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'sensitivity': float(results['sensitivity']),
                'specificity': float(results['specificity']),
                'roc_auc': float(roc_auc)
            },
            'classification_report': results['classification_report'],
            'evaluation_date': tf.timestamp().numpy().item()
        }
        
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(evaluation_data, f, indent=4)
        
        print("  ✓ Evaluation results saved to 'models/evaluation_results.json'")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Accuracy:     {results['accuracy']*100:.2f}%")
        print(f"Precision:    {results['precision']*100:.2f}%")
        print(f"Recall:       {results['recall']*100:.2f}%")
        print(f"F1-Score:     {results['f1_score']*100:.2f}%")
        print(f"Sensitivity:  {results['sensitivity']*100:.2f}%")
        print(f"Specificity:  {results['specificity']*100:.2f}%")
        print(f"ROC AUC:      {roc_auc:.4f}")
        print("=" * 60)

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path='models/best_model.h5')
    
    # Load model
    if not evaluator.load_model():
        print("\n✗ Failed to load model. Please train the model first.")
        exit(1)
    
    # Prepare test data
    test_gen = evaluator.prepare_test_data(batch_size=32)
    
    # Evaluate model
    results = evaluator.evaluate_model(test_gen)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(results['y_true'], results['y_pred'])
    
    # Plot ROC curve
    roc_auc = evaluator.plot_roc_curve(results['y_true'], results['predictions'])
    
    # Visualize predictions
    evaluator.visualize_predictions(test_gen, num_samples=9)
    
    # Save results
    evaluator.save_evaluation_results(results, roc_auc)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run '5_prediction.py' to test with custom images")