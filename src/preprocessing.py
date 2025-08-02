import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class LungCancerDataPreprocessor:
    def __init__(self, data_path="data", img_size=(224, 224)):
        """
        Initialize the lung cancer data preprocessor.
        
        Args:
            data_path (str): Path to the data directory
            img_size (tuple): Target image size (width, height)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Class mapping for different directory structures
        self.class_mapping = {
            'adenocarcinoma': 'adenocarcinoma',
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'adenocarcinoma',
            'large.cell.carcinoma': 'large.cell.carcinoma',
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'large.cell.carcinoma',
            'squamous.cell.carcinoma': 'squamous.cell.carcinoma',
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous.cell.carcinoma',
            'normal': 'normal'
        }
    
    def load_data_from_split(self, split='train'):
        """
        Load data from a specific split (train/test/valid).
        
        Args:
            split (str): Data split to load ('train', 'test', 'valid')
            
        Returns:
            tuple: (images, labels, class_names)
        """
        split_path = os.path.join(self.data_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist.")
            return [], [], []
        
        images = []
        labels = []
        class_names = []
        
        # Walk through the split directory
        for root, dirs, files in os.walk(split_path):
            if files and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                # Extract class name from directory path
                original_class_name = os.path.basename(root)
                
                # Map to standardized class name
                if original_class_name in self.class_mapping:
                    class_name = self.class_mapping[original_class_name]
                else:
                    # Try to extract class name from path
                    path_parts = root.split(os.sep)
                    for part in path_parts:
                        if part in self.class_mapping:
                            class_name = self.class_mapping[part]
                            break
                    else:
                        # Use original name if no mapping found
                        class_name = original_class_name
                
                if class_name not in class_names:
                    class_names.append(class_name)
                
                print(f"Loading {class_name} images from {original_class_name}...")
                
                # Load images for this class
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            # Load and preprocess image
                            img = self.load_and_preprocess_image(file_path)
                            if img is not None:
                                images.append(img)
                                labels.append(class_name)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        return np.array(images), np.array(labels), sorted(class_names)
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Create data generators for training and validation.
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        y_train_cat = self.label_encoder.fit_transform(y_train)
        y_val_cat = self.label_encoder.transform(y_val)
        
        # Convert to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train_cat, num_classes=len(self.class_names))
        y_val_cat = to_categorical(y_val_cat, num_classes=len(self.class_names))
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def visualize_data_distribution(self, labels, class_names):
        """
        Visualize the distribution of classes in the dataset.
        
        Args:
            labels: Array of labels
            class_names: List of class names
        """
        plt.figure(figsize=(10, 6))
        class_counts = [np.sum(labels == class_name) for class_name in class_names]
        
        plt.bar(class_names, class_counts, color='skyblue')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, count in enumerate(class_counts):
            plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total images: {len(labels)}")
        print(f"Class distribution: {dict(zip(class_names, class_counts))}")
    
    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            labels: Array of labels
            
        Returns:
            dict: Class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Convert labels to numerical format
        y_numerical = self.label_encoder.transform(labels)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_numerical),
            y=y_numerical
        )
        
        # Create dictionary
        class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
        
        return class_weight_dict

def load_data(train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=32):
    """
    Loads image data from train, validation, and test directories using Keras ImageDataGenerator.

    Args:
        train_dir (str): Path to training images.
        val_dir (str): Path to validation images.
        test_dir (str): Path to test images.
        image_size (tuple): Size to resize images to (default: (128, 128)).
        batch_size (int): Number of images per batch (default: 32).

    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        test_gen: Test data generator.
        class_indices: Mapping of class names to indices.
    """
    # Basic augmentation for training, only rescaling for val/test
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    class_indices = train_gen.class_indices

    return train_gen, val_gen, test_gen, class_indices 