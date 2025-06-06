"""
Model factory for creating different neural network architectures
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, applications
import config

def create_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create an improved Convolutional Neural Network (CNN) model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # Use a more efficient architecture with batch normalization
    inputs = layers.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Classification block
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use a standard optimizer with a fixed learning rate instead of a schedule
    # This allows the ReduceLROnPlateau callback to adjust the learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create an improved Deep Neural Network (DNN) model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # For DNN, we'll use a pretrained feature extractor to get better features
    # from the images before feeding them to dense layers
    inputs = layers.Input(shape=input_shape)
    
    # Use a simpler feature extraction approach to avoid compatibility issues
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_rnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create an improved Recurrent Neural Network (RNN) model with CNN features
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # For image classification, a pure RNN is not ideal
    # We'll use a CNN to extract features first, then process them with RNN
    height, width, channels = input_shape
    
    # Define input
    inputs = layers.Input(shape=input_shape)
    
    # CNN feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RNN (treating the feature maps as sequences)
    # After 2 max pooling layers, the spatial dimensions are reduced by a factor of 4
    new_height = height // 4
    new_width = width // 4
    x = layers.Reshape((new_height, new_width * 64))(x)
    
    # RNN layers
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    
    # Classification layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model(model_name, input_shape=None):
    """Factory function to get model by name"""
    if input_shape is None:
        input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        
    models_map = {
        'cnn': create_cnn,
        'dnn': create_dnn,
        'rnn': create_rnn
    }
    
    model_fn = models_map.get(model_name.lower())
    if not model_fn:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(models_map.keys())}")
    
    return model_fn(input_shape) 