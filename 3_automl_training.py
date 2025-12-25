"""
Breast Cancer Detection - Model Training Module
Implements transfer learning with ResNet18 architecture
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class BreastCancerModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build ResNet50-based transfer learning model"""
        print("\n[STEP 1] Building ResNet50 model...")
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("  ✓ Model built successfully")
        print(f"  Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_data_generators(self, batch_size=32):
        """Prepare data generators for training"""
        print("\n[STEP 2] Preparing data generators...")
        
        # Training data generator (with augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            'data/training',
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            'data/validation',
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            'data/testing',
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"  ✓ Training samples: {train_generator.samples}")
        print(f"  ✓ Validation samples: {val_generator.samples}")
        print(f"  ✓ Testing samples: {test_generator.samples}")
        print(f"  ✓ Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def train(self, train_gen, val_gen, epochs=50):
        """Train the model"""
        print("\n[STEP 3] Training model...")
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n  ✓ Training complete!")
        
        # Fine-tuning phase
        print("\n[STEP 4] Fine-tuning model (unfreezing last layers)...")
        
        # Unfreeze last 30 layers of base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key].extend(fine_tune_history.history[key])
        
        print("  ✓ Fine-tuning complete!")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training metrics"""
        print("\n[STEP 5] Plotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("  ✓ Training history saved to 'results/training_history.png'")
        plt.show()
    
    def save_model(self):
        """Save trained model and metadata"""
        print("\n[STEP 6] Saving model...")
        
        # Save full model
        self.model.save('models/breast_cancer_model.h5')
        
        # Save model weights
        self.model.save_weights('models/model_weights.h5')
        
        # Save metadata
        metadata = {
            'model_architecture': 'ResNet50',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_epochs': len(self.history.history['accuracy']),
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(self.history.history['val_accuracy']))
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("  ✓ Model saved successfully!")
        print(f"  Best validation accuracy: {metadata['best_val_accuracy']:.4f}")

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize model
    model_trainer = BreastCancerModel(input_shape=(224, 224, 3), num_classes=2)
    
    # Build model
    model_trainer.build_model()
    
    # Prepare data
    train_gen, val_gen, test_gen = model_trainer.prepare_data_generators(batch_size=32)
    
    # Train model
    model_trainer.train(train_gen, val_gen, epochs=50)
    
    # Plot results
    model_trainer.plot_training_history()
    
    # Save model
    model_trainer.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run '4_model_evaluation.py' to evaluate the model")