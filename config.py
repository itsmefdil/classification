"""
Configuration settings for the image classification application
"""

import os
from tensorflow.keras import mixed_precision

# --- Paths ---
DATA_DIR = 'tiktok_images'
OUTPUT_DIR = 'classification_results'
BAD_WORDS_FILE = 'kata_kotor.txt'
LOG_FILE = 'log_proses_klasifikasi.log'

# --- Model settings ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
TEST_SIZE = 0.2

# --- Mixed precision ---
USE_MIXED_PRECISION = True

# Apply mixed precision if enabled
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy('mixed_float16') 