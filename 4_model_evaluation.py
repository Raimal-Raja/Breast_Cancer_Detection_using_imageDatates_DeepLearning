"""
Breast Cancer Detection - Model Evaluation (FIXED)
Comprehensive evaluation with confusion matrix and sample predictions
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

print("=" * 80)
print("BREAST CANCER DETECTION - MODEL EVALUATION")
print("=" * 80)

# Load model
print("\n[STEP 1/6] Loading trained model...")
try:
    model = load_model('models/best_model.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first!")
    exit()

# Prepare test data
print("\n[STEP 2/6] Preparing test data...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Test samples: {test_gen.samples}")
print(f"   Classes: {test_gen.class_indices}")

# Get predictions
print("\n[STEP 3/6] Making predictions...")

predictions = model.predict(test_gen, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Calculate comprehensive metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"Overall Accuracy:  {accuracy*100:.2f}%")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-Score:          {f1:.4f}")
print("\nPer-Class Metrics:")
print(f"  Benign    - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}")
print(f"  Malignant - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}")
print("=" * 80)

# Classification report
print("\n[STEP 4/6] Detailed Classification Report:")
class_labels = ['benign', 'malignant']
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# Confusion Matrix
print("\n[STEP 5/6] Generating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)

# Calculate specificity and sensitivity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nConfusion Matrix Values:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")
print(f"\n  Sensitivity (TPR): {sensitivity*100:.2f}%")
print(f"  Specificity (TNR): {specificity*100:.2f}%")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={'label': 'Count'},
    ax=ax,
    square=True
)

ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)

# Add percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion matrix saved")
plt.show()

# ROC Curve
print("\n[STEP 6/6] Generating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_true, predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=150, bbox_inches='tight')
print(f"✅ ROC curve saved (AUC: {roc_auc:.4f})")
plt.show()

# Sample predictions visualization
print("\n[BONUS] Visualizing sample predictions...")

test_gen.reset()
batch_images, batch_labels = next(test_gen)
sample_predictions = model.predict(batch_images[:16], verbose=0)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Sample Test Predictions', fontsize=16, fontweight='bold')

for idx in range(16):
    row = idx // 4
    col = idx % 4
    
    img = batch_images[idx]
    true_label = class_labels[np.argmax(batch_labels[idx])]
    pred_label = class_labels[np.argmax(sample_predictions[idx])]
    confidence = np.max(sample_predictions[idx]) * 100
    
    # Probabilities
    prob_benign = sample_predictions[idx][0] * 100
    prob_malignant = sample_predictions[idx][1] * 100
    
    axes[row, col].imshow(img)
    
    # Color: green if correct, red if incorrect
    color = 'green' if true_label == pred_label else 'red'
    title = f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)\nB:{prob_benign:.0f}% M:{prob_malignant:.0f}%'
    axes[row, col].set_title(title, fontsize=9, color=color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Sample predictions saved")
plt.show()

# Calculate confidence distribution
print("\n[CONFIDENCE ANALYSIS]")
confidence_scores = np.max(predictions, axis=1) * 100
low_conf = np.sum(confidence_scores < 60)
med_conf = np.sum((confidence_scores >= 60) & (confidence_scores < 80))
high_conf = np.sum(confidence_scores >= 80)

print(f"Confidence distribution:")
print(f"  Low (<60%):      {low_conf} samples ({low_conf/len(confidence_scores)*100:.1f}%)")
print(f"  Medium (60-80%): {med_conf} samples ({med_conf/len(confidence_scores)*100:.1f}%)")
print(f"  High (>80%):     {high_conf} samples ({high_conf/len(confidence_scores)*100:.1f}%)")
print(f"  Average:         {np.mean(confidence_scores):.2f}%")

# Save comprehensive evaluation results
evaluation_data = {
    'test_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc)
    },
    'per_class_metrics': {
        'benign': {
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0])
        },
        'malignant': {
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1])
        }
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'confidence_stats': {
        'low_confidence': int(low_conf),
        'medium_confidence': int(med_conf),
        'high_confidence': int(high_conf),
        'average_confidence': float(np.mean(confidence_scores))
    }
}

with open('models/evaluation_results.json', 'w') as f:
    json.dump(evaluation_data, f, indent=4)

print("\n✅ Evaluation results saved")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE! ✅")
print("=" * 80)
print("\n📊 SUMMARY:")
print(f"   Test Accuracy:  {accuracy*100:.2f}%")
print(f"   Sensitivity:    {sensitivity*100:.2f}%")
print(f"   Specificity:    {specificity*100:.2f}%")
print(f"   ROC AUC:        {roc_auc:.4f}")

if accuracy < 0.6:
    print("\n⚠️  Low accuracy detected. Possible issues:")
    print("   - Insufficient training data")
    print("   - Model not distinguishing between classes")
    print("   - Data quality problems")
elif 100 * tn == test_gen.samples or 100 * tp == test_gen.samples:
    print("\n⚠️  Model predicting only one class!")
    print("   - Retrain with more balanced data")
    print("   - Check class weights")
else:
    print("\n✅ Model appears to be working correctly")

print("\n📋 NEXT STEP: Run '5_streamlit_app.py' to create the web app")
print("=" * 80)