"""
Data processing utilities for image classification
"""

import os
import logging
import numpy as np
import cv2
import hashlib
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import imagehash
from PIL import Image
import config
import time
import multiprocessing

def get_all_files_and_labels(directory, image_size, limit=None, remove_duplicates=True, 
                            duplicate_threshold=4, exact_duplicates_only=False):
    """
    Get all image files from a directory and extract labels based on text content
    
    Args:
        directory: Directory containing images
        image_size: Size to resize images to (height, width)
        limit: Maximum number of images to process
        remove_duplicates: Whether to remove duplicate images
        duplicate_threshold: Threshold for considering images as duplicates
        exact_duplicates_only: If True, only removes exact duplicates
        
    Returns:
        Tuple of (file_paths, labels, duplicate_groups)
    """
    start_time = time.time()
    logging.info(f"Scanning directory: {directory}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_paths.append(os.path.join(root, file))
    
    # Limit the number of files if requested
    if limit and len(file_paths) > limit:
        file_paths = file_paths[:limit]
    
    logging.info(f"Found {len(file_paths)} images")
    
    # Load bad words
    bad_words = []
    try:
        with open(config.BAD_WORDS_FILE, 'r') as f:
            bad_words = [line.strip().lower() for line in f if line.strip()]
        logging.info(f"Loaded {len(bad_words)} bad words from {config.BAD_WORDS_FILE}")
    except Exception as e:
        logging.error(f"Error loading bad words file: {e}")
    
    # Process images and detect text
    logging.info("Processing images and detecting text...")
    
    # Determine the optimal number of workers based on CPU count and mode
    if config.FORCE_CPU:
        max_workers = config.CPU_THREADS if config.CPU_THREADS > 0 else min(os.cpu_count() or 4, 4)
    else:
        max_workers = os.cpu_count() or 8
    
    # Create a ThreadPoolExecutor for parallel processing
    results = []
    
    # Process images in batches to avoid memory issues
    batch_size = 100
    num_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    duplicate_groups = []
    processed_files = []
    file_hashes = {}
    perceptual_hashes = {}
    
    # Initialize counters for logging
    total_processed = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_paths))
        batch_files = file_paths[start_idx:end_idx]
        
        logging.info(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} images)")
        
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(process_image, file_path, image_size, bad_words): file_path 
                for file_path in batch_files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        has_bad_word, image_hash, p_hash = result
                        batch_results.append((file_path, has_bad_word, image_hash, p_hash))
                        total_processed += 1
                        
                        # Log progress periodically
                        if total_processed % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = total_processed / elapsed if elapsed > 0 else 0
                            logging.info(f"Processed {total_processed}/{len(file_paths)} images ({rate:.2f} images/sec)")
                    else:
                        logging.warning(f"Failed to process image: {file_path}")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
        
        # Process duplicates for this batch
        if remove_duplicates:
            for file_path, has_bad_word, image_hash, p_hash in batch_results:
                # Check for exact duplicates first
                if image_hash in file_hashes:
                    # Found an exact duplicate
                    duplicate_file = file_hashes[image_hash]
                    
                    # Find or create a duplicate group for this hash
                    group_found = False
                    for group in duplicate_groups:
                        if duplicate_file in group:
                            group.append(file_path)
                            group_found = True
                            break
                    
                    if not group_found:
                        duplicate_groups.append([duplicate_file, file_path])
                else:
                    file_hashes[image_hash] = file_path
                    
                    # If we're not only looking for exact duplicates, check perceptual hashes
                    if not exact_duplicates_only and p_hash is not None:
                        # Check for similar images
                        is_duplicate = False
                        for existing_hash, existing_file in perceptual_hashes.items():
                            # Calculate Hamming distance
                            distance = p_hash - existing_hash
                            if distance <= duplicate_threshold:
                                # Found a similar image
                                # Find or create a duplicate group
                                group_found = False
                                for group in duplicate_groups:
                                    if existing_file in group:
                                        group.append(file_path)
                                        group_found = True
                                        break
                                
                                if not group_found:
                                    duplicate_groups.append([existing_file, file_path])
                                
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            perceptual_hashes[p_hash] = file_path
                            processed_files.append((file_path, has_bad_word))
                    else:
                        processed_files.append((file_path, has_bad_word))
        else:
            # If not removing duplicates, just add all files
            for file_path, has_bad_word, _, _ in batch_results:
                processed_files.append((file_path, has_bad_word))
        
        results.extend(batch_results)
    
    # If we're removing duplicates, log information about duplicates
    if remove_duplicates and duplicate_groups:
        duplicate_count = sum(len(group) for group in duplicate_groups) - len(duplicate_groups)
        logging.info(f"Found {duplicate_count} duplicate images in {len(duplicate_groups)} groups")
        
        # Log details of duplicate groups
        log_file = os.path.join(config.OUTPUT_DIR, 'duplicate_images.log')
        with open(log_file, 'w') as f:
            f.write(f"Total duplicate images: {duplicate_count}\n")
            f.write(f"Total duplicate groups: {len(duplicate_groups)}\n\n")
            
            for i, group in enumerate(duplicate_groups):
                f.write(f"Group {i+1} ({len(group)} images):\n")
                for file in group:
                    f.write(f"  {file}\n")
                f.write("\n")
        
        logging.info(f"Duplicate image details saved to {log_file}")
    
    # Extract file paths and labels
    if remove_duplicates:
        file_paths, labels = zip(*processed_files) if processed_files else ([], [])
    else:
        file_paths = [r[0] for r in results]
        labels = [r[1] for r in results]
    
    # Convert labels to integers (0 for clean, 1 for bad word)
    labels = [1 if label else 0 for label in labels]
    
    elapsed = time.time() - start_time
    logging.info(f"Processed {len(file_paths)} images in {elapsed:.2f} seconds ({len(file_paths)/elapsed:.2f} images/sec)")
    
    return file_paths, labels, duplicate_groups

def process_image(file_path, image_size, bad_words):
    """
    Process a single image file
    
    Args:
        file_path: Path to the image file
        image_size: Size to resize images to (height, width)
        bad_words: List of bad words to check for
        
    Returns:
        Tuple of (has_bad_word, image_hash, perceptual_hash)
    """
    try:
        # Calculate MD5 hash of the file for exact duplicate detection
        with open(file_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        
        # Load image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            logging.warning(f"Failed to load image: {file_path}")
            return None
        
        # Calculate perceptual hash for near-duplicate detection
        try:
            pil_img = Image.open(file_path)
            p_hash = imagehash.phash(pil_img)
        except Exception as e:
            logging.warning(f"Failed to calculate perceptual hash for {file_path}: {e}")
            p_hash = None
        
        # Resize image for OCR
        height, width = image_size
        img = cv2.resize(img, (width, height))
        
        # Extract text using OCR
        # Use simpler OCR settings for CPU-only mode to improve performance
        if config.FORCE_CPU:
            text = pytesseract.image_to_string(img, config='--psm 6')
        else:
            text = pytesseract.image_to_string(img)
        
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        
        # Check for bad words
        has_bad_word = False
        found_bad_words = []
        
        for word in bad_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text):
                has_bad_word = True
                found_bad_words.append(word)
        
        # Log bad word detections
        if has_bad_word:
            log_dir = config.OUTPUT_DIR
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            log_file = os.path.join(log_dir, 'bad_word_images.log')
            with open(log_file, 'a') as f:
                f.write(f"[{file_path}]\n")
                f.write(f"Found bad words: {', '.join(found_bad_words)}\n")
                f.write(f"Extracted text: {text}\n\n")
        
        return has_bad_word, image_hash, p_hash
        
    except Exception as e:
        logging.error(f"Error processing image {file_path}: {e}")
        return None

def count_bad_word_images(labels):
    """
    Count the number of images with bad words
    
    Args:
        labels: List of labels (0 for clean, 1 for bad word)
        
    Returns:
        Tuple of (bad_word_count, clean_count)
    """
    bad_word_count = sum(labels)
    clean_count = len(labels) - bad_word_count
    
    return bad_word_count, clean_count

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