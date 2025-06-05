import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import config

def generate_results_png(results, output_dir=config.OUTPUT_DIR, total_images=None, content_stats=None):
    """
    Generate PNG files with visualization of model comparison results
    
    Args:
        results: List of dictionaries with model evaluation results
        output_dir: Directory to save the output files
        total_images: Total number of images used in training
        content_stats: Dictionary with content analysis statistics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"📁 Created output directory: {output_dir}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Check if we have train/test size information
    has_split_info = 'Train Size' in df.columns and 'Test Size' in df.columns
    
    # Get train/test sizes if available
    if has_split_info:
        train_size = df['Train Size'].iloc[0]  # Should be the same for all models
        test_size = df['Test Size'].iloc[0]
        
        # Create a clean DataFrame for display (without train/test sizes)
        display_df = df[['Model', 'F1 Score', 'Accuracy', 'Loss']].copy()
    else:
        display_df = df.copy()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Add dataset information
    if total_images:
        info_text = f'Total Images: {total_images}'
        if has_split_info:
            info_text += f' (Train: {train_size}, Test: {test_size})'
        fig.text(0.5, 0.96, info_text, ha='center', fontsize=12)
    
    # 1. Bar plot for all metrics
    ax1 = axes[0, 0]
    metrics = ['F1 Score', 'Accuracy', 'Loss']
    x = np.arange(len(display_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        if metric == 'Loss':
            # Invert loss for better visualization (lower is better)
            values = 1 - display_df[metric]
            label = f'{metric} (Inverted)'
        else:
            values = display_df[metric]
            label = metric
        ax1.bar(x + i*width, values, width, label=label)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(display_df['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(display_df['Model'], display_df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Content distribution pie chart (if content stats available)
    ax3 = axes[1, 0]
    if content_stats and 'bad_word_images' in content_stats and 'clean_images' in content_stats:
        bad_count = content_stats['bad_word_images']
        clean_count = content_stats['clean_images']
        
        if bad_count + clean_count > 0:  # Avoid division by zero
            labels = ['Images with Bad Words', 'Clean Images']
            sizes = [bad_count, clean_count]
            colors = ['#FF7675', '#74B9FF']
            explode = (0.1, 0)  # explode the 1st slice (bad word images)
            
            ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax3.set_title('Content Distribution')
            
            # Add count annotation
            ax3.text(0, -1.2, f"Bad Word Images: {bad_count}", ha='center', fontsize=10)
            ax3.text(0, -1.4, f"Clean Images: {clean_count}", ha='center', fontsize=10)
    else:
        # If no content stats, show accuracy comparison
        bars = ax3.bar(display_df['Model'], display_df['Accuracy'], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
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
    bars = ax4.bar(display_df['Model'], display_df['Loss'], color=['#FF7675', '#74B9FF', '#A29BFE'])
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
    for _, row in display_df.iterrows():
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
    for i in range(len(display_df) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    title = 'Model Performance Summary Table'
    if total_images:
        title_info = f'Total: {total_images}'
        if has_split_info:
            title_info += f' (Train: {train_size}, Test: {test_size})'
        title += f' ({title_info})'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Save the table
    table_path = os.path.join(output_dir, 'results_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    logging.info(f"📊 Results table saved: {table_path}")
    
    # Create a separate content distribution chart if content stats available
    if content_stats and 'bad_word_images' in content_stats and 'clean_images' in content_stats:
        bad_count = content_stats['bad_word_images']
        clean_count = content_stats['clean_images']
        
        if bad_count + clean_count > 0:  # Avoid division by zero
            plt.figure(figsize=(10, 8))
            labels = ['Images with Bad Words', 'Clean Images']
            sizes = [bad_count, clean_count]
            colors = ['#FF7675', '#74B9FF']
            explode = (0.1, 0)  # explode the 1st slice (bad word images)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Content Distribution', fontsize=16, fontweight='bold')
            
            # Add count annotation
            plt.figtext(0.5, 0.01, f"Bad Word Images: {bad_count}, Clean Images: {clean_count}", 
                       ha='center', fontsize=12)
            
            # Save the content distribution chart
            content_path = os.path.join(output_dir, 'content_distribution.png')
            plt.savefig(content_path, dpi=300, bbox_inches='tight')
            logging.info(f"📊 Content distribution chart saved: {content_path}")
    
    # Create a text file with summary information
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"TRAINING SUMMARY\n")
        f.write(f"===============\n\n")
        
        # Dataset Information Section
        f.write(f"DATASET INFORMATION\n")
        f.write(f"-------------------\n\n")
        
        if total_images:
            f.write(f"Total images used: {total_images}\n\n")
            
            if has_split_info:
                f.write(f"Data Split:\n")
                f.write(f"  • Training images: {train_size}\n")
                f.write(f"  • Testing images: {test_size}\n")
                f.write(f"  • Split ratio: {train_size/total_images:.1%} train, {test_size/total_images:.1%} test\n\n")
        
        # Content Analysis Section
        if content_stats and 'bad_word_images' in content_stats and 'clean_images' in content_stats:
            bad_count = content_stats['bad_word_images']
            clean_count = content_stats['clean_images']
            
            f.write(f"CONTENT ANALYSIS\n")
            f.write(f"----------------\n\n")
            f.write(f"Content Distribution:\n")
            f.write(f"  • Images with bad words: {bad_count} ({bad_count/total_images:.1%} of total)\n")
            f.write(f"  • Clean images: {clean_count} ({clean_count/total_images:.1%} of total)\n\n")
            
        # Model Performance Section
        f.write(f"MODEL PERFORMANCE\n")
        f.write(f"----------------\n\n")
        f.write(f"{display_df.to_string(index=False)}\n\n")
        
        # Best Models Section
        f.write(f"BEST MODELS\n")
        f.write(f"-----------\n\n")
        f.write(f"• Best model by F1 Score: {display_df.loc[display_df['F1 Score'].idxmax()]['Model']} ({display_df['F1 Score'].max():.3f})\n")
        f.write(f"• Best model by Accuracy: {display_df.loc[display_df['Accuracy'].idxmax()]['Model']} ({display_df['Accuracy'].max():.3f})\n")
        f.write(f"• Best model by Loss: {display_df.loc[display_df['Loss'].idxmin()]['Model']} ({display_df['Loss'].min():.3f})\n\n")
        
        # Generated Files Section
        f.write(f"GENERATED FILES\n")
        f.write(f"--------------\n\n")
        f.write(f"• Model comparison chart: model_comparison.png\n")
        f.write(f"• Results table: results_table.png\n")
        if content_stats and 'bad_word_images' in content_stats:
            f.write(f"• Content distribution chart: content_distribution.png\n")
        f.write(f"• HTML report: training_report.html\n")
    
    logging.info(f"📄 Training summary saved: {summary_path}")
    
    plt.close('all')  # Close all figures to free memory
    
    return output_dir 