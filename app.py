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

def setup_logging(log_file=config.LOG_FILE):
    """Setup logging configuration"""
    # Ensure the output directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
        
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
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
    parser.add_argument('--keep-duplicates', action='store_true', help='Keep duplicate images')
    parser.add_argument('--duplicate-threshold', type=int, default=config.DUPLICATE_THRESHOLD, 
                        help='Threshold for considering images as duplicates (0-10, lower is stricter)')
    parser.add_argument('--exact-duplicates-only', action='store_true', default=config.EXACT_DUPLICATES_ONLY,
                        help='Only remove exact duplicates (same MD5 hash)')
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Update config.OUTPUT_DIR with args.output_dir if provided
    if args.output_dir != config.OUTPUT_DIR:
        config.OUTPUT_DIR = args.output_dir
        # Update LOG_FILE path as well
        config.LOG_FILE = os.path.join(config.OUTPUT_DIR, 'training.log')
    
    # Create output directory and setup logging
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")
    
    setup_logging()
    
    # Get image data with duplicate handling
    image_size = tuple(args.image_size)
    logging.info(f"Processing images from {args.image_dir} with size {image_size}")
    
    # Determine if we should remove duplicates
    remove_duplicates = not args.keep_duplicates
    if remove_duplicates:
        logging.info(f"Duplicate detection enabled (threshold: {args.duplicate_threshold}, exact only: {args.exact_duplicates_only})")
    else:
        logging.info("Duplicate detection disabled")
    
    file_paths, labels, duplicate_groups = get_all_files_and_labels(
        args.image_dir, 
        image_size, 
        args.limit,
        remove_duplicates=remove_duplicates,
        duplicate_threshold=args.duplicate_threshold,
        exact_duplicates_only=args.exact_duplicates_only
    )
    
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
    
    # Add duplicate information if available
    if duplicate_groups:
        duplicate_count = sum(len(group) for group in duplicate_groups) - len(duplicate_groups)
        content_stats['duplicate_groups'] = len(duplicate_groups)
        content_stats['duplicate_images'] = duplicate_count
    
    # Generate visualizations with total image count and content analysis
    output_directory = generate_results_png(results, args.output_dir, total_images, content_stats)
    
    # Generate HTML report if not disabled
    if not args.no_report:
        report_path = generate_html_report(results, args.output_dir, total_images, content_stats)
        logging.info(f"HTML report generated at: {report_path}")
    
    # Display results in console
    df = pd.DataFrame(results)
    display_df = df[['Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'Loss']].copy()
    
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
    
    # Print duplicate information if available
    if duplicate_groups and remove_duplicates:
        duplicate_count = sum(len(group) for group in duplicate_groups) - len(duplicate_groups)
        original_count = total_images + duplicate_count
        print(f"\nDuplicate detection settings:")
        print(f"  - Threshold: {args.duplicate_threshold} (lower is stricter)")
        print(f"  - Exact duplicates only: {'Yes' if args.exact_duplicates_only else 'No'}")
        print(f"\nDuplicate analysis:")
        print(f"  - Original dataset size: {original_count} images")
        print(f"  - Duplicate images removed: {duplicate_count} ({duplicate_count/original_count:.1%})")
        print(f"  - Duplicate groups: {len(duplicate_groups)}")
        print(f"  - See {os.path.join(args.output_dir, 'duplicate_images.log')} for details")
    
    # Print content analysis
    print("\n" + "="*60)
    print("CONTENT ANALYSIS")
    print("="*60)
    print(f"Images with bad words: {bad_word_count} ({bad_word_percent:.1f}%)")
    print(f"Clean images: {clean_count} ({clean_percent:.1f}%)")
    
    # Print bad word image log location if any bad words were found
    if bad_word_count > 0:
        bad_words_log_file = os.path.join(output_directory, 'bad_word_images.log')
        if os.path.exists(bad_words_log_file):
            print(f"\nDetailed log of images with bad words saved to:")
            print(f"  {bad_words_log_file}")
            print("\nSummary of bad word images:")
            
            # Read and display a short summary from the log file
            try:
                with open(bad_words_log_file, 'r') as f:
                    lines = f.readlines()
                    image_count = 0
                    bad_words_found = set()
                    
                    for line in lines:
                        if line.startswith('['):  # New image entry
                            image_count += 1
                        elif line.startswith('Found bad words:'):
                            words = line.strip().replace('Found bad words: ', '').split(', ')
                            for word in words:
                                bad_words_found.add(word)
                    
                    print(f"  - Total images with bad words: {image_count}")
                    if bad_words_found:
                        print(f"  - Bad words detected: {', '.join(sorted(bad_words_found))}")
            except Exception as e:
                logging.error(f"Error reading bad words log: {e}")
    
    # Print model performance
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(display_df.to_string(index=False))
    
    # Print best model information
    best_f1_model = display_df.loc[display_df['F1 Score'].idxmax()]['Model']
    best_acc_model = display_df.loc[display_df['Accuracy'].idxmax()]['Model']
    best_prec_model = display_df.loc[display_df['Precision'].idxmax()]['Model']
    best_recall_model = display_df.loc[display_df['Recall'].idxmax()]['Model']
    best_loss_model = display_df.loc[display_df['Loss'].idxmin()]['Model']
    
    print("\n" + "="*60)
    print("BEST MODELS")
    print("="*60)
    print(f"Best model by F1 Score: {best_f1_model} ({display_df['F1 Score'].max():.3f})")
    print(f"Best model by Accuracy: {best_acc_model} ({display_df['Accuracy'].max():.3f})")
    print(f"Best model by Precision: {best_prec_model} ({display_df['Precision'].max():.3f})")
    print(f"Best model by Recall: {best_recall_model} ({display_df['Recall'].max():.3f})")
    print(f"Best model by Loss: {best_loss_model} ({display_df['Loss'].min():.3f})")
    
    # Print output files
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"Output directory: {output_directory}")
    print(f"Model comparison chart: {os.path.join(output_directory, 'model_comparison.png')}")
    print(f"Results table: {os.path.join(output_directory, 'results_table.png')}")
    print(f"Content distribution chart: {os.path.join(output_directory, 'content_distribution.png')}")
    if duplicate_groups and remove_duplicates:
        print(f"Duplicate analysis chart: {os.path.join(output_directory, 'duplicate_analysis.png')}")
        print(f"Duplicate images log: {os.path.join(output_directory, 'duplicate_images.log')}")
    print(f"Training summary: {os.path.join(output_directory, 'training_summary.txt')}")
    # Add training history plots
    print(f"CNN training history: {os.path.join(output_directory, 'CNN_training_history.png')}")
    print(f"DNN training history: {os.path.join(output_directory, 'DNN_training_history.png')}")
    print(f"RNN training history: {os.path.join(output_directory, 'RNN_training_history.png')}")
    print(f"CNN model: {os.path.join(output_directory, 'CNN_best_model.keras')}")
    print(f"DNN model: {os.path.join(output_directory, 'DNN_best_model.keras')}")
    print(f"RNN model: {os.path.join(output_directory, 'RNN_best_model.keras')}")
    if not args.no_report:
        print(f"HTML report: {os.path.join(output_directory, 'training_report.html')}")
    print("="*60)

if __name__ == "__main__":
    main() 