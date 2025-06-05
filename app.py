import argparse
import logging
import os
import config
from utils.data_processing import get_all_files_and_labels, count_bad_word_images
from utils.visualization import generate_results_png
from utils.reporting import generate_html_report
from models.model_factory import create_cnn, create_dnn, create_rnn
from utils.training import train_and_evaluate
import pandas as pd

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image classification with text detection')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of images to process')
    parser.add_argument('--image-dir', type=str, default=config.DATA_DIR, help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR, help='Directory for results')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--image-size', type=int, nargs=2, default=list(config.IMAGE_SIZE), help='Image size (height width)')
    parser.add_argument('--no-report', action='store_true', help='Skip generating HTML report')
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Parse arguments and setup logging
    args = parse_arguments()
    setup_logging()
    
    # Get image data
    image_size = tuple(args.image_size)
    logging.info(f"Processing images from {args.image_dir} with size {image_size}")
    file_paths, labels = get_all_files_and_labels(args.image_dir, image_size, args.limit)
    
    # Calculate total images used
    total_images = len(file_paths)
    logging.info(f"Total images for training and testing: {total_images}")
    
    # Count images with bad words
    bad_word_count, clean_count = count_bad_word_images(labels)
    logging.info(f"Content analysis: {bad_word_count} images with bad words, {clean_count} clean images")
    
    # Calculate percentages
    if total_images > 0:
        bad_word_percent = (bad_word_count / total_images) * 100
        clean_percent = (clean_count / total_images) * 100
        logging.info(f"Content distribution: {bad_word_percent:.1f}% bad word images, {clean_percent:.1f}% clean images")
    
    # Run model training and evaluation
    results = [
        train_and_evaluate(create_cnn, 'CNN', file_paths, labels, image_size, args.batch_size, args.epochs),
        train_and_evaluate(create_dnn, 'DNN', file_paths, labels, image_size, args.batch_size, args.epochs),
        train_and_evaluate(create_rnn, 'RNN', file_paths, labels, image_size, args.batch_size, args.epochs)
    ]
    
    # Add content analysis to results metadata
    content_stats = {
        'total_images': total_images,
        'bad_word_images': bad_word_count,
        'clean_images': clean_count
    }
    
    # Generate visualizations with total image count and content analysis
    output_directory = generate_results_png(results, args.output_dir, total_images, content_stats)
    
    # Generate HTML report if not disabled
    if not args.no_report:
        report_path = generate_html_report(results, args.output_dir, total_images, content_stats)
        logging.info(f"HTML report generated at: {report_path}")
    
    # Display results in console
    df = pd.DataFrame(results)
    display_df = df[['Model', 'F1 Score', 'Accuracy', 'Loss']].copy()
    
    # Print dataset information
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total images: {total_images}")
    if 'Train Size' in df.columns and 'Test Size' in df.columns:
        train_size = df['Train Size'].iloc[0]
        test_size = df['Test Size'].iloc[0]
        print(f"Training images: {train_size} ({train_size/total_images:.1%})")
        print(f"Testing images: {test_size} ({test_size/total_images:.1%})")
    
    # Print content analysis
    print("\n" + "="*60)
    print("CONTENT ANALYSIS")
    print("="*60)
    print(f"Images with bad words: {bad_word_count} ({bad_word_percent:.1f}%)")
    print(f"Clean images: {clean_count} ({clean_percent:.1f}%)")
    
    # Print model performance
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(display_df.to_string(index=False))
    
    # Print best model information
    best_f1_model = display_df.loc[display_df['F1 Score'].idxmax()]['Model']
    best_acc_model = display_df.loc[display_df['Accuracy'].idxmax()]['Model']
    best_loss_model = display_df.loc[display_df['Loss'].idxmin()]['Model']
    
    print("\n" + "="*60)
    print("BEST MODELS")
    print("="*60)
    print(f"Best model by F1 Score: {best_f1_model} ({display_df['F1 Score'].max():.3f})")
    print(f"Best model by Accuracy: {best_acc_model} ({display_df['Accuracy'].max():.3f})")
    print(f"Best model by Loss: {best_loss_model} ({display_df['Loss'].min():.3f})")
    
    # Print output files
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"Output directory: {output_directory}")
    print(f"Model comparison chart: {os.path.join(output_directory, 'model_comparison.png')}")
    print(f"Results table: {os.path.join(output_directory, 'results_table.png')}")
    print(f"Content distribution chart: {os.path.join(output_directory, 'content_distribution.png')}")
    print(f"Training summary: {os.path.join(output_directory, 'training_summary.txt')}")
    if not args.no_report:
        print(f"HTML report: {os.path.join(output_directory, 'training_report.html')}")
    print("="*60)

if __name__ == "__main__":
    main() 