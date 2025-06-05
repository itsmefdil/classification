import os
import numpy as np
import pytesseract
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
import argparse
from tensorflow.keras import mixed_precision

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description='Image classification with text detection')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of images to process')
args = parser.parse_args()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log_proses_klasifikasi.log"),
        logging.StreamHandler()
    ]
)

# --- Setup Mixed Precision (optional) ---
mixed_precision.set_global_policy('mixed_float16')

# --- Load daftar kata kotor ---
def load_bad_words(filepath='kata_kotor.txt'):
    with open(filepath, 'r') as f:
        return set(line.strip().lower() for line in f)

kata_kotor = load_bad_words()

def contains_bad_word(text):
    words = set(text.lower().split())
    result = bool(kata_kotor.intersection(words))
    logging.info(f"Teks hasil OCR: \"{text.strip()}\" → {'Ditemukan kata kotor' if result else 'Tidak ditemukan kata kotor'}")
    return int(result)

# --- Fungsi untuk load data jadi generator agar hemat RAM ---
def data_generator(data_dir, image_size=(224, 224), batch_size=16):
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

# --- Ambil semua file untuk train/test split ---
def get_all_files_and_labels(data_dir, image_size=(224, 224), limit=None):
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

# --- Dataset dir ---
image_dir = 'tiktok_images'
image_size = (224, 224) 

# --- Ambil semua paths dan label ---
file_paths, labels = get_all_files_and_labels(image_dir, image_size, args.limit)

# --- Split data ---
train_paths, test_paths, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# --- Fungsi load dan preprocess image (perbaikan decode error) ---
def load_and_preprocess_image(path):
    def _load_image(path_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        img = Image.open(path_str).convert('RGB').resize(image_size)
        return np.array(img, dtype=np.uint8)
    
    img = tf.py_function(_load_image, [path], tf.uint8)
    img.set_shape([image_size[0], image_size[1], 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def tf_load_and_preprocess(path, label):
    img = load_and_preprocess_image(path)
    return img, label

# --- Buat tf.data.Dataset untuk train dan test ---
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
train_ds = train_ds.shuffle(len(train_paths)).map(tf_load_and_preprocess).batch(16).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, y_test))
test_ds = test_ds.map(tf_load_and_preprocess).batch(16).prefetch(tf.data.AUTOTUNE)

# --- Model CNN, DNN, RNN ---
input_shape = (image_size[0], image_size[1], 3)

def create_cnn():
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # float32 untuk kompatibilitas loss
    model = models.Model(inputs, outputs)
    return model

def create_dnn():
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = models.Model(inputs, outputs)
    return model

def create_rnn():
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], input_shape[1]*input_shape[2]))(inputs)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = models.Model(inputs, outputs)
    return model

# --- Pelatihan dan evaluasi ---
def train_and_evaluate(model_fn, name):
    logging.info(f"🔧 Memulai pelatihan model {name}...")
    model = model_fn()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
    logging.info(f"✅ Model {name} selesai dilatih.")

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
    return {'Model': name, 'F1 Score': f1, 'Accuracy': acc, 'Loss': loss}

# --- MISSING FUNCTION - Generate results as PNG files ---
def generate_results_png(results, output_dir):
    """
    Generate PNG files with visualization of model comparison results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"📁 Created output directory: {output_dir}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Bar plot for all metrics
    ax1 = axes[0, 0]
    metrics = ['F1 Score', 'Accuracy', 'Loss']
    x = np.arange(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        if metric == 'Loss':
            # Invert loss for better visualization (lower is better)
            values = 1 - df[metric]
            label = f'{metric} (Inverted)'
        else:
            values = df[metric]
            label = metric
        ax1.bar(x + i*width, values, width, label=label)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(df['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Model'], df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Accuracy comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(df['Model'], df['Accuracy'], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax3.set_title('Accuracy Comparison')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 4. Loss comparison
    ax4 = axes[1, 1]
    bars = ax4.bar(df['Model'], df['Loss'], color=['#FF7675', '#74B9FF', '#A29BFE'])
    ax4.set_title('Loss Comparison (Lower is Better)')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    logging.info(f"📊 Model comparison chart saved: {comparison_path}")
    
    # Create a summary table plot
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # Create table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['F1 Score']:.3f}",
            f"{row['Accuracy']:.3f}",
            f"{row['Loss']:.3f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Model', 'F1 Score', 'Accuracy', 'Loss'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#f0f0f0']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    
    # Save the table
    table_path = os.path.join(output_dir, 'results_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    logging.info(f"📊 Results table saved: {table_path}")
    
    plt.close('all')  # Close all figures to free memory
    
    return output_dir

# --- Jalankan semua model ---
results = [
    train_and_evaluate(create_cnn, 'CNN'),
    train_and_evaluate(create_dnn, 'DNN'),
    train_and_evaluate(create_rnn, 'RNN')
]

# --- Generate results as PNG files ---
output_directory = generate_results_png(results, 'classification_results')

# --- Also display results in console ---
df = pd.DataFrame(results)
print("\n" + "="*50)
print("HASIL EVALUASI MODEL")
print("="*50)
print(df)