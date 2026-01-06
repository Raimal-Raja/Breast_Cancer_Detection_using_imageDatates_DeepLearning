"""
Breast Cancer Detection - Model Training
Uses EfficientNetB3 with strong augmentation for high accuracy
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

print("=" * 80)
print("BREAST CANCER DETECTION - MODEL TRAINING")
print("=" * 80)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️  Using CPU (training will be slower)")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Build model
print("\n[STEP 1/5] Building EfficientNetB3 model...")

base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tuning strategy: freeze early layers, unfreeze later ones
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.2)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print(f"✅ Model built successfully")
print(f"   Total parameters: {model.count_params():,}")
trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
print(f"   Trainable parameters: {trainable:,}")

# Step 2: Data augmentation
print("\n[STEP 2/5] Preparing data with strong augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.35,
    height_shift_range=0.35,
    shear_range=0.35,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    channel_shift_range=40.0,
    fill_mode='reflect'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = val_test_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Data generators ready")
print(f"   Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
print(f"   Classes: {train_gen.class_indices}")

# Step 3: Class weights
print("\n[STEP 3/5] Computing class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Class weights: {class_weight_dict}")

# Step 4: Callbacks
print("\n[STEP 4/5] Setting up callbacks...")

checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    min_delta=0.0005,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-8,
    verbose=1
)

print("✅ Callbacks configured")

# Step 5: Training
print("\n[STEP 5/5] Training model...")
print("⏳ This will take 15-25 minutes (depending on hardware)...")
print("\n" + "=" * 80)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

print("\n" + "=" * 80)
print("✅ Training complete!")

# Load best model
try:
    model = tf.keras.models.load_model('models/best_model.keras')
    print("✅ Best model loaded from checkpoint")
except:
    print("⚠️  Using final model (checkpoint failed)")

# Evaluate on test set
print("\n[EVALUATION] Testing on held-out test set...")
test_results = model.evaluate(test_gen, verbose=1)
test_loss = test_results[0]
test_acc = test_results[1]
test_prec = test_results[2]
test_rec = test_results[3]
test_auc = test_results[4]

print(f"\n📊 TEST SET RESULTS:")
print(f"   Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision: {test_prec:.4f}")
print(f"   Recall:    {test_rec:.4f}")
print(f"   AUC:       {test_auc:.4f}")

# Check generalization
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = abs(train_acc - val_acc) * 100

print(f"\n   Final train accuracy: {train_acc*100:.2f}%")
print(f"   Final val accuracy:   {val_acc*100:.2f}%")
print(f"   Train-Val gap:        {gap:.2f}%")

if gap < 5:
    print("   ✅ Excellent generalization")
elif gap < 10:
    print("   ✅ Good generalization")
else:
    print("   ⚠️  Some overfitting detected")

# Visualization
print("\n[VISUALIZATION] Creating training history plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2.5, color='#2E86AB')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2.5, color='#A23B72')
axes[0, 0].axhline(y=test_acc, color='#F18F01', linestyle='--', linewidth=2, label=f'Test ({test_acc*100:.1f}%)')
axes[0, 0].set_title('Accuracy', fontweight='bold', fontsize=14)
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2.5, color='#2E86AB')
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2.5, color='#A23B72')
axes[0, 1].set_title('Loss', fontweight='bold', fontsize=14)
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Loss', fontsize=11)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# AUC
axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2.5, color='#2E86AB')
axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2.5, color='#A23B72')
axes[1, 0].axhline(y=test_auc, color='#F18F01', linestyle='--', linewidth=2, label=f'Test ({test_auc:.3f})')
axes[1, 0].set_title('AUC Score', fontweight='bold', fontsize=14)
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('AUC', fontsize=11)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Precision vs Recall
axes[1, 1].plot(history.history['precision'], label='Precision (Train)', linewidth=2.5, color='#2E86AB')
axes[1, 1].plot(history.history['recall'], label='Recall (Train)', linewidth=2.5, color='#06A77D')
axes[1, 1].plot(history.history['val_precision'], label='Precision (Val)', linewidth=2, color='#A23B72', linestyle='--')
axes[1, 1].plot(history.history['val_recall'], label='Recall (Val)', linewidth=2, color='#D81159', linestyle='--')
axes[1, 1].set_title('Precision & Recall', fontweight='bold', fontsize=14)
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training plots saved: results/training_history.png")
plt.close()

# Save metadata
metadata = {
    'model_architecture': 'EfficientNetB3',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_epochs': len(history.history['accuracy']),
    'final_train_accuracy': float(train_acc),
    'final_val_accuracy': float(val_acc),
    'test_accuracy': float(test_acc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'test_auc': float(test_auc),
    'train_val_gap': float(gap),
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
    'training_samples': train_gen.samples,
    'validation_samples': val_gen.samples,
    'test_samples': test_gen.samples
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# Save preprocessing config for inference
preprocessing_config = {
    'image_size': (224, 224),
    'preprocessing_steps': [
        'Grayscale conversion',
        'Resize to 224x224',
        'CLAHE enhancement',
        'Light denoising',
        'Normalization'
    ],
    'clahe_params': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},
    'denoise_params': {'h': 5, 'templateWindowSize': 7, 'searchWindowSize': 21}
}

with open('models/preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f, indent=4)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 FINAL RESULTS:")
print(f"   Test Accuracy:  {test_acc*100:.2f}%")
print(f"   Test AUC:       {test_auc:.4f}")
print(f"   Precision:      {test_prec:.4f}")
print(f"   Recall:         {test_rec:.4f}")

if test_acc >= 0.85 and test_auc >= 0.85:
    print("\n🎉 EXCELLENT! Model exceeds performance targets!")
elif test_acc >= 0.75:
    print("\n✅ GOOD! Model meets minimum performance requirements")
else:
    print("\n⚠️  Model performance below target - consider more data or training")

print("\n💾 Saved files:")
print("   • models/best_model.keras (Keras model)")
print("   • models/model_metadata.json")
print("   • models/preprocessing_config.json")
print("   • results/training_history.png")

print("\n📋 NEXT STEPS:")
print("   1. Run '4_model_evaluation.py' for detailed evaluation")
print("   2. Run '5_streamlit_app.py' to create the web interface")
print("   3. Download 'models/' folder to use locally")
print("\n" + "=" * 80)