import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def load_trained_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

def process_image(img_path, target_size=(224, 224)):
    """
    Process a single image for prediction.
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size for the image
        
    Returns:
        numpy.ndarray: Processed image array
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model, img_path, class_labels, target_size=(224, 224)):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained Keras model
        img_path (str): Path to the image file
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target size for the image
        
    Returns:
        tuple: (predicted_label, confidence)
    """
    img_array = process_image(img_path, target_size)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    return class_labels[predicted_class_index], confidence

def predict_batch(model, img_paths, class_labels, target_size=(224, 224)):
    """
    Predict classes for multiple images.
    
    Args:
        model: Trained Keras model
        img_paths (list): List of image file paths
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target size for the images
        
    Returns:
        list: List of prediction results
    """
    results = []
    for img_path in img_paths:
        try:
            label, confidence = predict_image(model, img_path, class_labels, target_size)
            results.append({'image': img_path, 'label': label, 'confidence': confidence})
        except Exception as e:
            results.append({'image': img_path, 'error': str(e)})
    return results

def predict_from_directory(model, directory_path, class_labels, target_size=(224, 224)):
    """
    Predict classes for all images in a directory.
    
    Args:
        model: Trained Keras model
        directory_path (str): Path to directory containing images
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target size for the images
        
    Returns:
        list: List of prediction results
    """
    img_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, file))
    
    return predict_batch(model, img_paths, class_labels, target_size)

def get_lung_cancer_class_labels():
    """
    Get the class labels for lung cancer classification.
    
    Returns:
        dict: Mapping of class indices to labels
    """
    return {
        0: 'adenocarcinoma',
        1: 'large.cell.carcinoma', 
        2: 'squamous.cell.carcinoma',
        3: 'normal'
    }

if __name__ == "__main__":
    # Example usage for lung cancer classification
    model_path = 'models/best_model.h5'
    
    # Lung cancer class labels
    class_labels = get_lung_cancer_class_labels()
    
    # Example image paths (you would replace these with actual paths)
    img_paths = [
        'data/test/adenocarcinoma/sample1.png',
        'data/test/normal/sample2.png'
    ]
    
    try:
        # Load the trained model
        model = load_trained_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Make predictions
        results = predict_batch(model, img_paths, class_labels)
        
        # Print results
        for result in results:
            if 'label' in result:
                print(f"📊 {result['image']}: {result['label']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"❌ Error predicting {result['image']}: {result['error']}")
                
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_model.py")
    except Exception as e:
        print(f"❌ Error: {e}") 