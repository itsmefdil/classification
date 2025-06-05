import os
import numpy as np
import pytesseract
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import config

# --- Load daftar kata kotor ---
def load_bad_words(filepath=config.BAD_WORDS_FILE):
    with open(filepath, 'r') as f:
        return set(line.strip().lower() for line in f)

kata_kotor = load_bad_words()

def contains_bad_word(text):
    words = set(text.lower().split())
    result = bool(kata_kotor.intersection(words))
    logging.info(f"Teks hasil OCR: \"{text.strip()}\" → {'Ditemukan kata kotor' if result else 'Tidak ditemukan kata kotor'}")
    return int(result)

# --- Fungsi untuk load data jadi generator agar hemat RAM ---
def data_generator(data_dir, image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE):
    images, labels = [], []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(data_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB').resize(image_size)
                text = pytesseract.image_to_string(img)
                label = contains_bad_word(text)
                img_array = np.array(img, dtype=np.float16) / 255.0
                img.close()

                images.append(img_array)
                labels.append(label)

                if len(images) == batch_size:
                    yield np.array(images), np.array(labels)
                    images, labels = [], []

            except Exception as e:
                logging.error(f"Gagal memproses {fname}: {e}")

    if images:
        yield np.array(images), np.array(labels)

# --- Get all files and labels ---
def get_all_files_and_labels(data_dir, image_size=config.IMAGE_SIZE, limit=None):
    file_paths = []
    labels = []
    for i, fname in enumerate(os.listdir(data_dir)):
        if limit is not None and i >= limit:
            logging.info(f"Reached image limit ({limit}). Stopping processing.")
            break
            
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(data_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB').resize(image_size)
                text = pytesseract.image_to_string(img)
                label = contains_bad_word(text)
                img.close()

                file_paths.append(img_path)
                labels.append(label)

            except Exception as e:
                logging.error(f"Gagal memproses {fname}: {e}")
    
    logging.info(f"Processed {len(file_paths)} images")
    return file_paths, labels

# --- Count images with bad words ---
def count_bad_word_images(labels):
    """Count the number of images containing bad words"""
    if not labels:
        return 0, 0
    
    bad_word_count = sum(labels)
    clean_count = len(labels) - bad_word_count
    
    logging.info(f"Images with bad words: {bad_word_count}, Clean images: {clean_count}")
    return bad_word_count, clean_count

# --- Fungsi load dan preprocess image ---
def load_and_preprocess_image(path, image_size):
    def _load_image(path_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        img = Image.open(path_str).convert('RGB').resize(image_size)
        return np.array(img, dtype=np.uint8)
    
    img = tf.py_function(_load_image, [path], tf.uint8)
    img.set_shape([image_size[0], image_size[1], 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def tf_load_and_preprocess(path, label, image_size):
    img = load_and_preprocess_image(path, image_size)
    return img, label

# --- Create TensorFlow datasets ---
def create_tf_datasets(file_paths, labels, image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE, test_size=config.TEST_SIZE):
    # Split data
    train_paths, test_paths, y_train, y_test = train_test_split(
        file_paths, labels, test_size=test_size, random_state=42
    )
    
    # Create tf.data.Dataset for train and test
    def _preprocess_wrapper(path, label):
        return tf_load_and_preprocess(path, label, image_size)
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
    train_ds = train_ds.shuffle(len(train_paths)).map(_preprocess_wrapper).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, y_test))
    test_ds = test_ds.map(_preprocess_wrapper).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, y_test 