import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import logging
import warnings
import config
from utils.data_processing import create_tf_datasets

def train_and_evaluate(model_fn, name, file_paths, labels, image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS):
    """Train and evaluate a model"""
    logging.info(f"🔧 Memulai pelatihan model {name}...")
    
    # Create datasets
    train_ds, test_ds, y_test = create_tf_datasets(
        file_paths, labels, image_size=image_size, batch_size=batch_size
    )
    
    # Get dataset sizes
    train_size = len(file_paths) - len(y_test)
    test_size = len(y_test)
    
    logging.info(f"Dataset split: {train_size} training images, {test_size} testing images")
    
    # Create and compile model
    input_shape = (image_size[0], image_size[1], 3)
    model = model_fn(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(train_ds, epochs=epochs, validation_data=test_ds, verbose=1)
    logging.info(f"✅ Model {name} selesai dilatih.")

    # Evaluate model
    y_pred_prob = model.predict(test_ds).flatten()
    y_true = np.array(y_test)

    y_pred = (y_pred_prob > 0.5).astype("int32")
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logging.warning(f"⚠️ y_test hanya mengandung 1 kelas: {unique_labels}. Beberapa metrik mungkin tidak valid.")
        labels = [0, 1]
    else:
        labels = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred_prob, labels=[0, 1])

    logging.info(f"📊 Evaluasi {name} → F1: {f1:.3f}, Accuracy: {acc:.3f}, Loss: {loss:.3f}")
    
    # Return results with dataset sizes
    return {
        'Model': name, 
        'F1 Score': f1, 
        'Accuracy': acc, 
        'Loss': loss,
        'Train Size': train_size,
        'Test Size': test_size
    } 