import tensorflow as tf
from tensorflow.keras import layers, models
import config

def create_cnn(input_shape=None):
    """Create a Convolutional Neural Network model"""
    if input_shape is None:
        input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # float32 for compatibility with loss
    model = models.Model(inputs, outputs)
    return model

def create_dnn(input_shape=None):
    """Create a Deep Neural Network model"""
    if input_shape is None:
        input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = models.Model(inputs, outputs)
    return model

def create_rnn(input_shape=None):
    """Create a Recurrent Neural Network model"""
    if input_shape is None:
        input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], input_shape[1]*input_shape[2]))(inputs)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = models.Model(inputs, outputs)
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