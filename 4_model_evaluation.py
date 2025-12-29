"""
Breast Cancer Detection - Model Evaluation for Google Colab
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
print("\n[STEP 1/5] Loading trained model...")
try:
    model = load_model('models/best_model.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first!")
    exit()

# Prepare test data
print("\n[STEP 2/5] Preparing test data...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/testing',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Test samples: {test_generator.samples}")

# Evaluate model
print("\n[STEP 3/5] Evaluating model...")

predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Calculate sensitivity and specificity
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print(f"Accuracy:     {accuracy*100:.2f}%")
print(f"Precision:    {precision*100:.2f}%")
print(f"Recall:       {recall*100:.2f}%")
print(f"F1-Score:     {f1*100:.2f}%")
print(f"Sensitivity:  {sensitivity*100:.2f}%")
print(f"Specificity:  {specificity*100:.2f}%")
print("=" * 80)

# Detailed classification report
print("\n[CLASSIFICATION REPORT]")
class_labels = ['benign', 'malignant']
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# Step 4: Plot confusion matrix
print("\n[STEP 4/5] Generating confusion matrix...")

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
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
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Confusion matrix saved")
plt.show()

# Step 5: Plot ROC curve
print("\n[STEP 5/5] Generating ROC curve...")

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
plt.title('ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=150, bbox_inches='tight')
print(f"✅ ROC curve saved (AUC: {roc_auc:.4f})")
plt.show()

# Visualize sample predictions
print("\n[BONUS] Visualizing sample predictions...")

test_generator.reset()
batch_images, batch_labels = next(test_generator)
sample_predictions = model.predict(batch_images[:9], verbose=0)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

for idx in range(9):
    row = idx // 3
    col = idx % 3
    
    img = batch_images[idx]
    true_label = class_labels[np.argmax(batch_labels[idx])]
    pred_label = class_labels[np.argmax(sample_predictions[idx])]
    confidence = np.max(sample_predictions[idx]) * 100
    
    axes[row, col].imshow(img)
    
    color = 'green' if true_label == pred_label else 'red'
    title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
    axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Sample predictions saved")
plt.show()

# Save evaluation results
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
    'confusion_matrix': cm.tolist(),
    'evaluation_date': tf.timestamp().numpy().item()
}

with open('models/evaluation_results.json', 'w') as f:
    json.dump(evaluation_data, f, indent=4)

print("\n✅ Evaluation results saved")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE! ✅")
print("=" * 80)
print("\n📋 NEXT STEP: Run '5_launch_app.py' to start the Streamlit dashboard")
print("=" * 80)