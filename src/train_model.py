import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data
from model import build_model, train_model

def main():
    """
    Main training script for lung cancer classification.
    """
    print("🚀 Starting Lung Cancer Classification Training")
    print("=" * 50)
    
    # Configuration
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    IMAGE_SIZE = (224, 224)  # Increased size for better performance
    BATCH_SIZE = 32
    NUM_CLASSES = 4  # adenocarcinoma, large.cell.carcinoma, squamous.cell.carcinoma, normal
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    print(f"📁 Data directories:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Validation: {VAL_DIR}")
    print(f"  Test: {TEST_DIR}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of classes: {NUM_CLASSES}")
    
    # Step 1: Load data
    print("\n📊 Loading data...")
    try:
        train_gen, val_gen, test_gen, class_indices = load_data(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        
        print("✅ Data loaded successfully!")
        print(f"  Training samples: {train_gen.samples}")
        print(f"  Validation samples: {val_gen.samples}")
        print(f"  Test samples: {test_gen.samples}")
        print(f"  Classes: {list(class_indices.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Build and train model
    print("\n🤖 Building and training model...")
    try:
        # Build model
        model = build_model(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            num_classes=NUM_CLASSES
        )
        
        print("✅ Model built successfully!")
        print(f"  Model parameters: {model.count_params():,}")
        
        # Train model
        trained_model, history = train_model(
            train_gen=train_gen,
            val_gen=val_gen,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            num_classes=NUM_CLASSES,
            model_path='models/best_model.h5'
        )
        
        print("✅ Model trained successfully!")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Step 3: Evaluate model
    print("\n📈 Evaluating model...")
    try:
        # Get predictions
        predictions = trained_model.predict(test_gen)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = list(class_indices.keys())
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        return
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: models/best_model.h5")
    print(f"📊 Confusion matrix saved to: models/confusion_matrix.png")
    print(f"📈 Training history saved to: models/training_history.png")

if __name__ == "__main__":
    main()

def retrain_model():
    """
    Retrain an existing model with new data.
    """
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)  # Ensure eager execution is on
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import Precision, Recall, AUC
    
    MODEL_PATH = "models/best_model.h5"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/valid"  # Updated to match your directory structure
    TEST_DIR = "data/test"
    IMAGE_SIZE = (224, 224)  # Updated to match your training size
    BATCH_SIZE = 32
    EPOCHS = 2
    
    print("🔄 Starting Model Retraining")
    print("=" * 40)
    
    try:
        print("📥 Loading existing model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")

        # Recompile with a new optimizer instance
        print("🔧 Recompiling model...")
        model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        print("✅ Model recompiled successfully!")

        print("📊 Loading data...")
        train_gen, val_gen, test_gen, class_indices = load_data(
            TRAIN_DIR, VAL_DIR, TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
        )
        print("✅ Data loaded successfully!")
        print(f"  Training samples: {train_gen.samples}")
        print(f"  Validation samples: {val_gen.samples}")
        print(f"  Classes: {list(class_indices.keys())}")
        
        print("🚀 Retraining model...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1
        )
        print("✅ Retraining completed!")
        
        print("💾 Saving updated model...")
        model.save(MODEL_PATH)
        print("✅ Model saved successfully!")
        
        print("🎉 Retraining complete!")
        
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first using the main training function")
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "retrain":
        retrain_model()
    else:
    main() 