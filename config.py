"""
Configuration settings for the image classification application
"""

import os
import datetime
from tensorflow.keras import mixed_precision

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'tiktok_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
BAD_WORDS_FILE = os.path.join(BASE_DIR, 'kata_kotor.txt')
LOG_FILE = os.path.join(OUTPUT_DIR, 'training.log')

# --- Model parameters ---
IMAGE_SIZE = (224, 224)  # height, width
BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.2

# --- Duplicate detection parameters ---
REMOVE_DUPLICATES = True  # Default behavior is to remove duplicates
DUPLICATE_THRESHOLD = 4   # Hamming distance threshold (0-10, lower is stricter)
EXACT_DUPLICATES_ONLY = False  # If True, only removes exact duplicates (same MD5 hash)

# --- Mixed precision ---
USE_MIXED_PRECISION = True

# Apply mixed precision if enabled
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy('mixed_float16') 