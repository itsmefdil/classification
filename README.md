# Image Classification with Text Detection

This project performs image classification using text detection to identify inappropriate content in images.

## Project Structure

```
.
├── app.py                  # Main application entry point
├── config.py               # Configuration settings
├── kata_kotor.txt          # List of inappropriate words
├── models/                 # Model definitions
│   ├── __init__.py
│   └── model_factory.py    # Model creation functions
└── utils/                  # Utility functions
    ├── __init__.py
    ├── data_processing.py  # Data loading and processing
    ├── training.py         # Model training and evaluation
    └── visualization.py    # Result visualization
```

## Features

- Detects inappropriate text in images using OCR
- Trains and compares multiple model architectures (CNN, DNN, RNN)
- Generates visualizations of model performance
- Modular design for easy maintenance and extension

## Usage

Run the application with default settings:

```bash
python app.py
```

### Command Line Arguments

- `--limit`: Limit the number of images to process
- `--image-dir`: Directory containing images (default: 'tiktok_images')
- `--output-dir`: Directory for results (default: 'classification_results')
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 16)
- `--image-size`: Image size as height width (default: 224 224)

Example:

```bash
python app.py --limit 100 --epochs 10 --batch-size 32
```

## Configuration

Edit `config.py` to change default settings:

- Image size
- Batch size
- Number of epochs
- File paths
- Mixed precision settings

## Requirements

See `requirements.txt` for dependencies. 