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
    # Add new CPU-related arguments
    parser.add_argument('--force-cpu', action='store_true', default=config.FORCE_CPU,
                        help='Force CPU-only mode (no GPU)')
    parser.add_argument('--cpu-threads', type=int, default=config.CPU_THREADS,
                        help='Number of CPU threads to use (0 = auto-detect)')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
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
    
    # Update hardware acceleration settings
    config.FORCE_CPU = args.force_cpu
    config.CPU_THREADS = args.cpu_threads
    config.USE_MIXED_PRECISION = not args.no_mixed_precision
    
    # Create output directory and setup logging
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Created output directory: {config.OUTPUT_DIR}")
    
    setup_logging()
    
    # Log hardware settings
    if config.FORCE_CPU:
        logging.info(f"Running in CPU-only mode with {config.CPU_THREADS if config.CPU_THREADS > 0 else 'auto-detected'} threads")
    else:
        logging.info("GPU acceleration enabled")
    
    if config.USE_MIXED_PRECISION and not config.FORCE_CPU:
        logging.info("Mixed precision enabled")
    else:
        logging.info("Mixed precision disabled")
    
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
    
    # Print hardware settings
    print("\n" + "="*60)
    print("HARDWARE SETTINGS")
    print("="*60)
    print(f"CPU-only mode: {'Enabled' if config.FORCE_CPU else 'Disabled'}")
    if config.FORCE_CPU:
        print(f"CPU threads: {config.CPU_THREADS if config.CPU_THREADS > 0 else 'Auto-detected'}")
    print(f"Mixed precision: {'Enabled' if config.USE_MIXED_PRECISION and not config.FORCE_CPU else 'Disabled'}")
    
    # Print training times if available
    if 'Training Time' in df.columns:
        print("\nTraining Times:")
        for _, row in df.iterrows():
            print(f"  - {row['Model']}: {row['Training Time']}")
    
    # Print best model information
    best_f1_model = display_df.loc[display_df['F1 Score'].idxmax()]['Model']
    best_acc_model = display_df.loc[display_df['Accuracy'].idxmax()]['Model']
    best_prec_model = display_df.loc[display_df['Precision'].idxmax()]['Model']
    best_recall_model = display_df.loc[display_df['Recall'].idxmax()]['Model']
    
    print("\nBest performing models:")
    print(f"  - Best F1 Score: {best_f1_model}")
    print(f"  - Best Accuracy: {best_acc_model}")
    print(f"  - Best Precision: {best_prec_model}")
    print(f"  - Best Recall: {best_recall_model}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()