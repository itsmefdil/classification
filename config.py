"""
Configuration settings for the image classification application
"""

import os
import datetime
from tensorflow.keras import mixed_precision
from dotenv import load_dotenv
import platform

# Load environment variables from .env file
load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'tiktok_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
BAD_WORDS_FILE = os.path.join(BASE_DIR, 'kata_kotor.txt')
LOG_FILE = os.path.join(OUTPUT_DIR, 'training.log')

# --- Model parameters ---
IMAGE_SIZE = (int(os.getenv('IMAGE_SIZE_HEIGHT', 224)), int(os.getenv('IMAGE_SIZE_WIDTH', 224)))  # height, width
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 10))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))

# --- Duplicate detection parameters ---
REMOVE_DUPLICATES = os.getenv('REMOVE_DUPLICATES', 'true').lower() == 'true'
DUPLICATE_THRESHOLD = int(os.getenv('DUPLICATE_THRESHOLD', 4))
EXACT_DUPLICATES_ONLY = os.getenv('EXACT_DUPLICATES_ONLY', 'false').lower() == 'true'

# --- Hardware acceleration settings ---
USE_MIXED_PRECISION = os.getenv('USE_MIXED_PRECISION', 'true').lower() == 'true'
FORCE_CPU = os.getenv('FORCE_CPU', 'false').lower() == 'true'
CPU_THREADS = int(os.getenv('CPU_THREADS', 0))  # 0 means auto-detect

# Configure TensorFlow for CPU-only mode if requested
if FORCE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS if CPU_THREADS > 0 else os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(CPU_THREADS if CPU_THREADS > 0 else os.cpu_count())
    print(f"Running in CPU-only mode with {CPU_THREADS if CPU_THREADS > 0 else 'auto-detected'} threads")
else:
    # Check if we're on AMD hardware and optimize accordingly
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        if 'AMD' in cpu_info.get('brand_raw', ''):
            print("AMD CPU detected, optimizing for AMD architecture")
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    except ImportError:
        # If cpuinfo is not available, try to detect AMD from platform info
        if 'AMD' in platform.processor():
            print("AMD CPU detected, optimizing for AMD architecture")
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Apply mixed precision if enabled and not in CPU-only mode
if USE_MIXED_PRECISION and not FORCE_CPU:
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled") 