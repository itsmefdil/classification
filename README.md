# Image Classification Application

This application performs image classification with text detection to identify images containing inappropriate content.

## Features

- Image classification using multiple neural network architectures (CNN, DNN, RNN)
- Text detection to identify images with bad words
- Duplicate image detection and removal
- Performance comparison of different models
- Detailed HTML reports and visualizations
- Support for CPU-only mode with AMD optimization
- Configurable via environment variables or command-line arguments

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd classification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (optional):
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Usage

### Basic Usage

Run the application with default settings:

```bash
python main.py
```

### Command Line Arguments

- `--limit`: Limit the number of images to process
- `--image-dir`: Directory containing images (default: `tiktok_images`)
- `--output-dir`: Directory for results
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--image-size`: Image size as height width (e.g., `--image-size 224 224`)
- `--no-report`: Skip generating HTML report
- `--keep-duplicates`: Keep duplicate images
- `--duplicate-threshold`: Threshold for considering images as duplicates (0-10, lower is stricter)
- `--exact-duplicates-only`: Only remove exact duplicates (same MD5 hash)
- `--force-cpu`: Force CPU-only mode (no GPU)
- `--cpu-threads`: Number of CPU threads to use (0 = auto-detect)
- `--no-mixed-precision`: Disable mixed precision training

### Environment Variables

You can configure the application using environment variables in a `.env` file:

```
# Image size for processing
IMAGE_SIZE_HEIGHT=224
IMAGE_SIZE_WIDTH=224

# Training parameters
BATCH_SIZE=32
EPOCHS=10
TEST_SIZE=0.2

# Duplicate detection parameters
REMOVE_DUPLICATES=true
DUPLICATE_THRESHOLD=4
EXACT_DUPLICATES_ONLY=false

# Hardware acceleration settings
FORCE_CPU=false
CPU_THREADS=0
USE_MIXED_PRECISION=true
```

## CPU-Only Mode

For systems without a GPU or when you want to use CPU only:

```bash
python main.py --force-cpu
```

This mode automatically:
- Optimizes for AMD processors when detected
- Reduces model complexity to improve performance
- Adjusts batch sizes and thread counts for better CPU utilization
- Uses GRU instead of LSTM for better performance on CPU

For AMD processors, the application automatically enables AMD-specific optimizations.

## Output

The application generates the following outputs in the specified output directory:

- Training logs
- Model performance comparison charts
- Content distribution charts
- Duplicate image analysis (if enabled)
- Training history plots for each model
- Saved models
- HTML report with all results (if not disabled)

## Examples

Train with a smaller batch size and force CPU mode:
```bash
python main.py --batch-size 16 --force-cpu
```

Process only 100 images for testing:
```bash
python main.py --limit 100
```

Use a specific image directory and output directory:
```bash
python main.py --image-dir /path/to/images --output-dir ./results
```

## License

[License Information]

## Acknowledgements

[Any acknowledgements] 