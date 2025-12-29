"""
Breast Cancer Detection - Model Training for Google Colab
Trains ResNet50 model with GPU acceleration
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 80)
print("BREAST CANCER DETECTION - MODEL TRAINING")
print("=" * 80)

# Check GPU availability
print("\n[GPU CHECK]")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("⚠️  No GPU detected. Training will use CPU (slower)")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Build model
print("\n[STEP 1/5] Building ResNet50 model...")

# Load pre-trained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build custom model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(f"✅ Model built successfully")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")

# Step 2: Prepare data generators
print("\n[STEP 2/5] Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    'data/testing',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Testing samples: {test_generator.samples}")
print(f"✅ Classes: {train_generator.class_indices}")

# Step 3: Train model (Phase 1 - Frozen base)
print("\n[STEP 3/5] Training model (Phase 1 - Frozen base)...")

checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Initial training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

print("✅ Phase 1 training complete!")

# Step 4: Fine-tuning (Phase 2 - Unfreeze last layers)
print("\n[STEP 4/5] Fine-tuning model (Phase 2 - Unfreezing layers)...")

# Unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(f"   Trainable parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")

# Continue training
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Combine histories
for key in history.history.keys():
    history.history[key].extend(fine_tune_history.history[key])

print("✅ Phase 2 fine-tuning complete!")

# Step 5: Plot training history
print("\n[STEP 5/5] Plotting training history...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Training History', fontsize=16)

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training history saved to 'results/training_history.png'")
plt.show()

# Save model and metadata
print("\n[SAVING] Saving model and metadata...")

model.save('models/breast_cancer_model.h5')
model.save_weights('models/model_weights.h5')

metadata = {
    'model_architecture': 'ResNet50',
    'input_shape': [224, 224, 3],
    'num_classes': 2,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_epochs': len(history.history['accuracy']),
    'final_train_accuracy': float(history.history['accuracy'][-1]),
    'final_val_accuracy': float(history.history['val_accuracy'][-1]),
    'best_val_accuracy': float(max(history.history['val_accuracy'])),
    'gpu_used': len(gpus) > 0
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Model saved successfully!")

print("\n" + "=" * 80)
print("TRAINING COMPLETE! ✅")
print("=" * 80)
print(f"\n📊 RESULTS:")
print(f"   Final Training Accuracy: {metadata['final_train_accuracy']*100:.2f}%")
print(f"   Final Validation Accuracy: {metadata['final_val_accuracy']*100:.2f}%")
print(f"   Best Validation Accuracy: {metadata['best_val_accuracy']*100:.2f}%")
print("\n📋 NEXT STEP: Run '4_model_evaluation.py' to evaluate the model")
print("=" * 80)