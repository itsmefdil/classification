# Image Classification with Text Detection

This project performs image classification to detect inappropriate content in images using OCR (Optical Character Recognition) to extract text from images and identify bad words.

## Features

- **Text Detection**: Uses OCR to extract text from images
- **Content Classification**: Identifies images containing inappropriate words
- **Multiple Models**: Compares CNN, DNN, and RNN model performance
- **Duplicate Detection**: Identifies and removes duplicate or near-duplicate images
- **Comprehensive Reporting**: Generates HTML reports, charts, and detailed logs
- **Modular Architecture**: Well-organized code structure for maintainability

## Requirements

- Python 3.7+
- Tesseract OCR
- Dependencies listed in `requirements.txt`

## Installation

1. Install Tesseract OCR:
   - On macOS: `brew install tesseract`
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place images in the `data` directory
   - Create a `bad_words.txt` file with one word per line

## Usage

Basic usage:
```
python app.py
```

With options:
```
python app.py --image-dir /path/to/images --output-dir /path/to/output --limit 1000
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--image-dir` | Directory containing images (default: from config) |
| `--output-dir` | Directory for results (default: from config) |
| `--limit` | Limit the number of images to process |
| `--epochs` | Number of training epochs (default: 10) |
| `--batch-size` | Batch size for training (default: 32) |
| `--image-size` | Image size as height width (default: 224 224) |
| `--no-report` | Skip generating HTML report |
| `--keep-duplicates` | Keep duplicate images (don't remove them) |
| `--duplicate-threshold` | Threshold for considering images as duplicates (0-10, lower is stricter) |
| `--exact-duplicates-only` | Only remove exact duplicates |

### Duplicate Detection

The system can identify and remove duplicate or near-duplicate images to improve training efficiency:

1. **Exact Duplicates**: Images with identical MD5 hashes
2. **Near-Duplicates**: Visually similar images identified by perceptual hashing

Duplicate detection can be configured with:
- `--keep-duplicates`: Disable duplicate removal
- `--duplicate-threshold`: Set sensitivity (0-10, lower is stricter)
- `--exact-duplicates-only`: Only remove exact duplicates

A detailed log of detected duplicates is saved to `duplicate_images.log` in the output directory.

## Project Structure

```
.
├── app.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── data/                  # Directory for image data
├── models/                # Model definitions
│   ├── __init__.py
│   └── model_factory.py   # Factory for creating models
└── utils/                 # Utility modules
    ├── __init__.py
    ├── data_processing.py # Data loading and preprocessing
    ├── reporting.py       # HTML report generation
    ├── training.py        # Model training functions
    └── visualization.py   # Chart generation
```

## Output

The system generates several output files:
- HTML report with model performance and dataset analysis
- Charts comparing model performance
- Content distribution analysis
- Duplicate image analysis
- Detailed logs of the training process

## License

MIT License 