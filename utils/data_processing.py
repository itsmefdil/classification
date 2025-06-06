import os
import numpy as np
import pytesseract
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import config
import hashlib
import cv2
from collections import defaultdict
import datetime

# --- Load daftar kata kotor ---
def load_bad_words(filepath=config.BAD_WORDS_FILE):
    with open(filepath, 'r') as f:
        return set(line.strip().lower() for line in f)

kata_kotor = load_bad_words()

def contains_bad_word(text, image_path=None):
    words = set(text.lower().split())
    result = bool(kata_kotor.intersection(words))
    
    # Log the result with the image path if provided
    if image_path:
        if result:
            log_message = f"Ditemukan kata kotor dalam gambar: {image_path}"
            logging.info(log_message)
            # Also log to a special file for bad word images
            log_bad_word_image(image_path, text, kata_kotor.intersection(words))
        else:
            logging.info(f"Teks hasil OCR dari {os.path.basename(image_path)}: \"{text.strip()}\" → Tidak ditemukan kata kotor")
    else:
        logging.info(f"Teks hasil OCR: \"{text.strip()}\" → {'Ditemukan kata kotor' if result else 'Tidak ditemukan kata kotor'}")
    
    return int(result)

def log_bad_word_image(image_path, text, bad_words):
    """
    Log information about an image containing bad words to a dedicated log file
    
    Args:
        image_path: Path to the image file
        text: Text extracted from the image
        bad_words: Set of bad words found in the text
    """
    # Create the log directory if it doesn't exist
    log_dir = os.path.dirname(config.OUTPUT_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a log file specifically for bad word images
    bad_words_log_file = os.path.join(config.OUTPUT_DIR, 'bad_word_images.log')
    
    # Get absolute path for better tracking
    abs_path = os.path.abspath(image_path)
    
    # Format the log entry
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] Image: {abs_path}\n"
    log_entry += f"Found bad words: {', '.join(bad_words)}\n"
    log_entry += f"Extracted text: \"{text.strip()}\"\n"
    log_entry += "-" * 80 + "\n"
    
    # Write to the log file
    with open(bad_words_log_file, 'a') as f:
        f.write(log_entry)

# --- Calculate image hash for duplicate detection ---
def calculate_image_hash(image_path, hash_size=8):
    """
    Calculate perceptual hash for an image to detect near-duplicates
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size + 1))
        
        # Calculate difference hash
        diff = resized[:hash_size, :hash_size] > resized[1:, 1:]
        
        # Convert to hash string
        hash_value = sum([2 ** i for i, v in enumerate(diff.flatten()) if v])
        return hash_value
    except Exception as e:
        logging.error(f"Error calculating hash for {image_path}: {e}")
        return None

def calculate_file_hash(file_path):
    """
    Calculate MD5 hash of a file for exact duplicate detection
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating file hash for {file_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """
    Calculate Hamming distance between two hashes
    """
    return bin(hash1 ^ hash2).count('1')

def detect_duplicates(file_paths, threshold=config.DUPLICATE_THRESHOLD, exact_only=config.EXACT_DUPLICATES_ONLY):
    """
    Detect duplicate or near-duplicate images
    
    Args:
        file_paths: List of image file paths
        threshold: Maximum hamming distance to consider as duplicate
        exact_only: If True, only detect exact duplicates (same MD5 hash)
        
    Returns:
        duplicate_groups: List of lists, where each inner list contains paths of duplicate images
        unique_files: List of file paths that are not duplicates
    """
    # Calculate hashes for all images
    hashes = {}
    file_hashes = {}
    
    logging.info(f"Calculating image hashes for {len(file_paths)} files...")
    logging.info(f"Duplicate detection settings: threshold={threshold}, exact_only={exact_only}")
    
    for path in file_paths:
        # First check exact duplicates using MD5
        file_hash = calculate_file_hash(path)
        if file_hash:
            if file_hash in file_hashes:
                file_hashes[file_hash].append(path)
            else:
                file_hashes[file_hash] = [path]
            
        # Then calculate perceptual hash for similar images if not exact_only
        if not exact_only:
            img_hash = calculate_image_hash(path)
            if img_hash is not None:
                hashes[path] = img_hash
    
    # Find exact duplicates first (same MD5 hash)
    exact_duplicates = []
    for hash_val, paths in file_hashes.items():
        if len(paths) > 1:
            exact_duplicates.append(paths)
    
    # Find near-duplicates using perceptual hash if not exact_only
    near_duplicates = []
    if not exact_only:
        # Group images by hash bucket for faster comparison
        hash_buckets = defaultdict(list)
        for path, hash_val in hashes.items():
            # Use the most significant bits as bucket key
            bucket = hash_val & 0xFFFFF0  # Use first 20 bits
            hash_buckets[bucket].append(path)
        
        # Compare images within each bucket
        processed_paths = set()
        
        for bucket, paths in hash_buckets.items():
            for i, path1 in enumerate(paths):
                if path1 in processed_paths:
                    continue
                    
                group = [path1]
                for path2 in paths[i+1:]:
                    if path2 in processed_paths:
                        continue
                        
                    distance = hamming_distance(hashes[path1], hashes[path2])
                    if distance <= threshold:
                        group.append(path2)
                        processed_paths.add(path2)
                
                if len(group) > 1:
                    processed_paths.add(path1)
                    near_duplicates.append(group)
    
    # Combine exact and near duplicates
    all_duplicates = exact_duplicates + near_duplicates
    
    # Get unique files (not in any duplicate group)
    duplicate_files = set()
    for group in all_duplicates:
        for path in group:
            duplicate_files.add(path)
    
    unique_files = [path for path in file_paths if path not in duplicate_files]
    
    # Add one representative from each duplicate group
    for group in all_duplicates:
        unique_files.append(group[0])
    
    # Log results
    exact_dup_count = sum(len(group) for group in exact_duplicates) - len(exact_duplicates)
    near_dup_count = sum(len(group) for group in near_duplicates) - len(near_duplicates)
    
    logging.info(f"Found {len(duplicate_files)} duplicate images in {len(all_duplicates)} groups:")
    logging.info(f"  - Exact duplicates: {exact_dup_count} in {len(exact_duplicates)} groups")
    if not exact_only:
        logging.info(f"  - Near duplicates: {near_dup_count} in {len(near_duplicates)} groups")
    logging.info(f"Keeping {len(unique_files)} unique images")
    
    return all_duplicates, unique_files

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

# --- Get all files and labels with duplicate handling ---
def get_all_files_and_labels(data_dir, image_size=config.IMAGE_SIZE, limit=None, 
                             remove_duplicates=config.REMOVE_DUPLICATES, 
                             duplicate_threshold=config.DUPLICATE_THRESHOLD,
                             exact_duplicates_only=config.EXACT_DUPLICATES_ONLY):
    """
    Get all image files and their labels with duplicate detection
    
    Args:
        data_dir: Directory containing images
        image_size: Size to resize images to (height, width)
        limit: Maximum number of images to process
        remove_duplicates: Whether to remove duplicate images
        duplicate_threshold: Threshold for considering images as duplicates
        exact_duplicates_only: If True, only remove exact duplicates
        
    Returns:
        valid_paths: List of image file paths
        labels: List of labels (1 for images with bad words, 0 for clean images)
        duplicate_groups: List of lists, where each inner list contains paths of duplicate images
    """
    file_paths = []
    
    # Collect all image file paths
    for i, fname in enumerate(os.listdir(data_dir)):
        if limit is not None and i >= limit:
            logging.info(f"Reached image limit ({limit}). Stopping processing.")
            break
            
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(data_dir, fname)
            file_paths.append(img_path)
    
    # Handle duplicates if requested
    duplicate_groups = []
    if remove_duplicates and len(file_paths) > 1:
        logging.info("Checking for duplicate images...")
        duplicate_groups, file_paths = detect_duplicates(
            file_paths, 
            threshold=duplicate_threshold,
            exact_only=exact_duplicates_only
        )
        
        if duplicate_groups:
            duplicate_count = sum(len(g) for g in duplicate_groups) - len(duplicate_groups)
            logging.info(f"Removed {duplicate_count} duplicate images")
            
            # Save duplicate information to a log file
            duplicate_log_path = os.path.join(config.OUTPUT_DIR, 'duplicate_images.log')
            with open(duplicate_log_path, 'w') as f:
                f.write(f"Found {len(duplicate_groups)} groups of duplicate images:\n\n")
                for i, group in enumerate(duplicate_groups, 1):
                    f.write(f"Group {i} ({len(group)} images):\n")
                    # Keep the first image
                    f.write(f"  [KEPT] {group[0]}\n")
                    # Remove the rest
                    for path in group[1:]:
                        f.write(f"  [REMOVED] {path}\n")
                    f.write("\n")
    
    # Process images for OCR and labeling
    labels = []
    processed_paths = []
    
    logging.info(f"Processing {len(file_paths)} images for text detection...")
    
    for img_path in file_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            text = pytesseract.image_to_string(img)
            
            # Pass the image path to contains_bad_word
            label = contains_bad_word(text, img_path)
            
            labels.append(label)
            processed_paths.append(img_path)
            
        except Exception as e:
            logging.error(f"Failed to process {img_path}: {e}")
    
    logging.info(f"Processed {len(processed_paths)} images")
    
    return processed_paths, labels, duplicate_groups

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