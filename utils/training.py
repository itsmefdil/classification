"""
Training and evaluation functions for image classification models
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import logging
import config
import os
import matplotlib.pyplot as plt

def train_and_evaluate(model_creator, model_name, file_paths, labels, image_size, batch_size=32, epochs=10):
    """
    Train and evaluate a model
    
    Args:
        model_creator: Function to create the model
        model_name: Name of the model for logging
        file_paths: List of image file paths
        labels: List of labels (0 or 1)
        image_size: Size to resize images to (height, width)
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Dictionary with model performance metrics
    """
    logging.info(f"Training {model_name} model...")
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Split data into train and test sets with stratification
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=config.TEST_SIZE, random_state=42, stratify=labels
    )
    
    logging.info(f"Training set: {len(train_paths)} images")
    logging.info(f"Test set: {len(test_paths)} images")
    
    # Create TensorFlow datasets with data augmentation for training
    def preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    def augment_image(image):
        # Apply random augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image
    
    def load_and_preprocess_train(path, label):
        image = preprocess_image(path)
        image = augment_image(image)
        return image, label
    
    def load_and_preprocess_test(path, label):
        image = preprocess_image(path)
        return image, label
    
    # Create training dataset with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths))
    train_ds = train_ds.map(load_and_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create test dataset without augmentation
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(load_and_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    
    # Create and compile model
    model = model_creator(input_shape=(image_size[0], image_size[1], 3))
    
    # Create callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.OUTPUT_DIR, f"{model_name}_best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions on test set
    y_pred_probs = model.predict(test_ds)
    
    # Convert probabilities to class predictions
    if y_pred_probs.shape[-1] > 1:  # Multi-class case
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:  # Binary case
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    f1 = f1_score(test_labels, y_pred, average='weighted')
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(test_labels, y_pred, average='weighted', zero_division=0)
    
    # Get the final loss value
    loss = history.history['val_loss'][-1]
    
    # Log detailed results
    logging.info(f"{model_name} Results:")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  Loss: {loss:.4f}")
    
    # Log classification report
    report = classification_report(test_labels, y_pred)
    logging.info(f"Classification Report for {model_name}:\n{report}")
    
    # Plot training history
    plot_training_history(history, model_name, config.OUTPUT_DIR)
    
    # Return results as dictionary with additional metrics
    return {
        'Model': model_name,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Loss': loss,
        'Train Size': len(train_paths),
        'Test Size': len(test_paths)
    }

def plot_training_history(history, model_name, output_dir):
    """
    Plot training history and save to file
    
    Args:
        history: Training history from model.fit()
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'))
    plt.close() 