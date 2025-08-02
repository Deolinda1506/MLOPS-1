from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

def build_model(input_shape, num_classes):
    """
    Builds a CNN model for image classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model

def train_model(train_gen, val_gen, input_shape, num_classes,  class_weight=None, model_path='../models/best_model.h5'):
    """
    Trains the CNN model with the given data generators.

    Args:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        model_path (str): Where to save the best model.

    Returns:
        model (Sequential): Trained Keras model.
    """
    model = build_model(input_shape, num_classes)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight
    )

    return model, history 