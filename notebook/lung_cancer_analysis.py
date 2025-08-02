#!/usr/bin/env python3
"""
Lung Cancer Classification - Complete Analysis
Adapted from the user's code for lung cancer dataset
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('../src')

from preprocessing import load_data
from model import train_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def main():
    print("🔬 LUNG CANCER CLASSIFICATION ANALYSIS")
    print("=" * 50)
    
    # Configuration for lung cancer dataset
    train_dir = '../data/train'
    val_dir = '../data/valid'  # Note: using 'valid' not 'val'
    test_dir = '../data/test'
    image_size = (224, 224)  # Increased size for better performance
    batch_size = 32
    
    print("📊 Loading lung cancer dataset...")
    train_gen, val_gen, test_gen, class_indices = load_data(
        train_dir, val_dir, test_dir, image_size, batch_size
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Classes: {list(class_indices.keys())}")
    
    # Analyze class distribution
    labels = list(class_indices.keys())
    counts = [train_gen.labels.tolist().count(i) for i in range(len(labels))]
    
    print(f"\n📊 Class distribution: {dict(zip(labels, counts))}")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title("Class Distribution (Training Set)")
    plt.xlabel("Lung Cancer Types")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    # Display sample images
    x, y = next(train_gen)
    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(x[i])
        plt.title(labels[np.argmax(y[i])], fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze pixel intensity distribution
    sample_imgs, _ = next(train_gen)
    all_pixels = sample_imgs.flatten()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot([sample_imgs[:,:,:,i].flatten() for i in range(3)], 
               labels=['Red', 'Green', 'Blue'])
    plt.title('RGB Channel Distribution')
    plt.ylabel('Pixel Value')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate class weights for imbalanced dataset
    labels_numerical = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(labels_numerical), 
        y=labels_numerical
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n⚖️ Class weights for imbalanced dataset:")
    for i, weight in enumerate(class_weights):
        print(f"  {labels[i]}: {weight:.3f}")
    
    # Visualize class weights
    plt.figure(figsize=(8, 5))
    plt.bar(labels, class_weights, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Class Weights (Balanced)')
    plt.xlabel('Lung Cancer Types')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Train the model
    input_shape = (224, 224, 3)
    num_classes = len(class_indices)
    
    print("\n🚀 Starting model training...")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Using class weights: {class_weight_dict}")
    
    model, history = train_model(
        train_gen, val_gen, input_shape, num_classes, 
        class_weight=class_weight_dict
    )
    
    print("✅ Training completed!")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate on test set
    y_true = test_gen.classes
    y_pred = model.predict(test_gen)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Lung Cancer Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # Save training history
    os.makedirs('../models', exist_ok=True)
    with open('../models/history.json', 'w') as f:
        json.dump(history.history, f)
    
    print("✅ Results saved to ../models/history.json")
    print("🎉 Lung Cancer Classification Analysis Complete!")

if __name__ == "__main__":
    main() 