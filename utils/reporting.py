import os
import pandas as pd
import logging
import datetime
import config

def generate_html_report(results, output_dir=config.OUTPUT_DIR, total_images=None, content_stats=None):
    """
    Generate a detailed HTML report with training results
    
    Args:
        results: List of dictionaries with model evaluation results
        output_dir: Directory to save the output files
        total_images: Total number of images used in training
        content_stats: Dictionary with content analysis statistics
    """
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
        train_size = None
        test_size = None
    
    # Check if we have content stats
    has_content_stats = (content_stats is not None and 
                         'bad_word_images' in content_stats and 
                         'clean_images' in content_stats)
    
    if has_content_stats:
        bad_word_count = content_stats['bad_word_images']
        clean_count = content_stats['clean_images']
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification Training Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f9f9f9;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 30px;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 40px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                border-left: 5px solid #3498db;
            }}
            .summary {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .summary-box {{
                flex: 1;
                min-width: 200px;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary-box h3 {{
                margin-top: 0;
                color: #3498db;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }}
            .summary-box p {{
                margin: 10px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
            }}
            th, td {{
                padding: 15px 20px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .best {{
                font-weight: bold;
                color: #27ae60;
            }}
            .chart-container {{
                margin: 40px 0;
                text-align: center;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .chart-container p {{
                margin-top: 15px;
                color: #7f8c8d;
                font-style: italic;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .content-stats {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin: 30px 0;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            .content-stats-text {{
                flex: 1;
                padding-right: 20px;
            }}
            .content-stats-text p {{
                margin: 10px 0;
                line-height: 1.8;
            }}
            .content-stats-chart {{
                flex: 1;
                text-align: center;
            }}
            .bad-word {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .clean {{
                color: #2980b9;
                font-weight: bold;
            }}
            .progress-container {{
                width: 100%;
                background-color: #f1f1f1;
                border-radius: 5px;
                margin: 15px 0;
                overflow: hidden;
            }}
            .progress-bar {{
                height: 24px;
                border-radius: 0;
                text-align: center;
                line-height: 24px;
                color: white;
                font-weight: bold;
            }}
            .progress-bad {{
                background-color: #e74c3c;
            }}
            .progress-clean {{
                background-color: #2980b9;
            }}
            .section {{
                margin: 40px 0;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            .highlight-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
            .highlight-box h4 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .stat-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Image Classification Training Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <section class="section">
                <h2>Summary</h2>
                <div class="summary">
                    <div class="summary-box">
                        <h3>Dataset Information</h3>
    """
    
    if total_images:
        html_content += f"""
                        <p><strong>Total Images:</strong> <span class="stat-value">{total_images}</span></p>
        """
        if has_split_info:
            html_content += f"""
                        <div class="highlight-box">
                            <h4>Data Split:</h4>
                            <p><strong>Training Images:</strong> <span class="stat-value">{train_size}</span> ({train_size/total_images:.1%})</p>
                            <p><strong>Testing Images:</strong> <span class="stat-value">{test_size}</span> ({test_size/total_images:.1%})</p>
                        </div>
            """
    else:
        html_content += "<p>No dataset information available</p>"
    
    html_content += """
                    </div>
                    <div class="summary-box">
                        <h3>Best Performing Models</h3>
    """
    
    # Add best models
    best_f1_model = display_df.loc[display_df['F1 Score'].idxmax()]['Model']
    best_acc_model = display_df.loc[display_df['Accuracy'].idxmax()]['Model']
    best_loss_model = display_df.loc[display_df['Loss'].idxmin()]['Model']
    
    html_content += f"""
                        <p><strong>Best F1 Score:</strong> <span class="stat-value">{best_f1_model}</span> ({display_df['F1 Score'].max():.3f})</p>
                        <p><strong>Best Accuracy:</strong> <span class="stat-value">{best_acc_model}</span> ({display_df['Accuracy'].max():.3f})</p>
                        <p><strong>Best Loss:</strong> <span class="stat-value">{best_loss_model}</span> ({display_df['Loss'].min():.3f})</p>
                    </div>
                </div>
            </section>
    """
    
    # Add content analysis section if available
    if has_content_stats and total_images > 0:
        bad_percent = (bad_word_count / total_images) * 100
        clean_percent = (clean_count / total_images) * 100
        
        html_content += f"""
            <section class="section">
                <h2>Content Analysis</h2>
                <div class="content-stats">
                    <div class="content-stats-text">
                        <h3>Dataset Content Distribution</h3>
                        <p>The dataset contains:</p>
                        <p>
                            <span class="bad-word">{bad_word_count} images with bad words</span> 
                            ({bad_percent:.1f}% of total dataset)
                        </p>
                        <p>
                            <span class="clean">{clean_count} clean images</span> 
                            ({clean_percent:.1f}% of total dataset)
                        </p>
                        
                        <div class="progress-container">
                            <div class="progress-bar progress-bad" style="width: {bad_percent}%; float: left;">
                                {bad_percent:.1f}%
                            </div>
                            <div class="progress-bar progress-clean" style="width: {clean_percent}%; float: left;">
                                {clean_percent:.1f}%
                            </div>
                        </div>
                    </div>
                    <div class="content-stats-chart">
                        <img src="content_distribution.png" alt="Content Distribution" style="max-width: 300px;">
                    </div>
                </div>
            </section>
        """
    
    html_content += """
            <section class="section">
                <h2>Model Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>F1 Score</th>
                        <th>Accuracy</th>
                        <th>Loss</th>
                    </tr>
    """
    
    # Add model results
    for _, row in display_df.iterrows():
        model_name = row['Model']
        f1_class = "best" if model_name == best_f1_model else ""
        acc_class = "best" if model_name == best_acc_model else ""
        loss_class = "best" if model_name == best_loss_model else ""
        
        html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td class="{f1_class}">{row['F1 Score']:.3f}</td>
                        <td class="{acc_class}">{row['Accuracy']:.3f}</td>
                        <td class="{loss_class}">{row['Loss']:.3f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </section>
            
            <section class="section">
                <h2>Visualizations</h2>
                <div class="chart-container">
                    <h3>Model Performance Comparison</h3>
                    <img src="model_comparison.png" alt="Model Comparison Chart">
                    <p>Comparison of model performance across different metrics (F1 Score, Accuracy, and Loss)</p>
                </div>
                
                <div class="chart-container">
                    <h3>Performance Summary Table</h3>
                    <img src="results_table.png" alt="Results Table">
                    <p>Summary table of model performance metrics</p>
                </div>
            </section>
            
            <div class="footer">
                <p>Generated by Image Classification Training System</p>
                <p>© 2023 All Rights Reserved</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, 'training_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"📄 HTML training report saved: {report_path}")
    
    return report_path 