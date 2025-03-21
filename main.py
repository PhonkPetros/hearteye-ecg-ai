import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

###############################################################################
# Import ECGLoader, ECGCleaner, ECGExtractor from existing files
###############################################################################
from data_processing.loader import ECGLoader
from data_processing.cleaner import ECGCleaner
from feature_extraction.extractor import ECGExtractor
from visualization.plots import Plots  # Import Plots class from plots.py

###############################################################################
# main() implementation with cumulative analysis visualization
###############################################################################
def main():
    """
    Main function to:
      - Load data via ECGLoader
      - Process ECG records one by one
      - Maintain cumulative statistics as each record is processed
      - Periodically create/update visualizations showing the evolution of statistics
    """
 
    base_dir = r"C:\Users\prota\Desktop\hearteye-ecg-ai\data\external\mimic-iv-ecg-data\files"
    csv_path = r"C:\Users\prota\Desktop\hearteye-ecg-ai\data\external\mimic-iv-ecg-data\records_w_diag_icd10.csv"
    output_dir = r"C:\Users\prota\Desktop\hearteye-ecg-ai\data\mimic-iv-ecg-data\results"
    clean_data_dir = os.path.join(output_dir, "cleaned_data")
    visualization_dir = os.path.join(output_dir, "visualization")
    cumulative_dir = os.path.join(visualization_dir, "cumulative")

    features_file = os.path.join(output_dir, "ecg_features.csv")

    print("[INFO] Base directory:", base_dir)
    print("[INFO] CSV path:", csv_path)
    print("[INFO] Feature results will be saved to:", features_file)
    print("[INFO] Individual cleaned data files will be saved to:", clean_data_dir)
    print("[INFO] Visualizations will be saved to:", visualization_dir)
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clean_data_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(cumulative_dir, exist_ok=True)

    # Create ECGLoader and optionally load the CSV
    loader = ECGLoader(base_dir=base_dir, csv_path=csv_path)
    df_labels = loader.load_csv_labels()  # Only if you need label info
    if df_labels is not None:
        print(f"[INFO] Loaded CSV: {len(df_labels)} rows, columns={list(df_labels.columns)}")

    # Prepare ECGCleaner and ECGExtractor
    cleaner = ECGCleaner(fs=500, lowcut=0.5, highcut=40, order=4, baseline_cutoff=0.5)
    extractor = ECGExtractor()

    # Initialize data structures for cumulative statistics
    record_count = 0
    feature_results = []
    
    # Dictionary to track cumulative averages for visualization
    cumulative_stats = {
        'record_counts': [],
        'QRS_duration_avg': [],
        'QT_duration_avg': [],
        'PQ_interval_avg': [],
        'P_duration_avg': [],
        'processing_batches': []  # To track when visualizations were generated
    }
    
    # Define checkpoints for visualization updates (exponentially increasing)
    visualization_checkpoints = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    next_checkpoint_idx = 0
    
    # Optional limit on total records to process
    max_records = None  # Set to None to process all records
    
    # Process records one by one
    for hea_path, dat_path in loader.iter_ecg_records():
        record_count += 1
        print(f"\n[RECORD #{record_count}] => .hea: {hea_path}, .dat: {dat_path}")
        
        # Optional limit on records
        if max_records and record_count > max_records:
            print(f"[INFO] Reached maximum record count of {max_records}. Stopping.")
            break

        record_base = os.path.splitext(hea_path)[0]
        record_name = os.path.basename(record_base)
        
        try:
            signals, fields = loader.load_ecg_record(record_base)
        except RuntimeError as e:
            print(f"[WARNING] Could not load record {hea_path}: {e}")
            continue

        fs = fields.get("fs", 500)
        
        # If multiple channels, pick the first
        if signals.ndim == 2 and signals.shape[1] > 1:
            raw_signal = signals[:, 0]
        else:
            raw_signal = signals
        
        # Clean ECG
        try:
            ecg_cleaned = cleaner.clean_ecg(raw_signal)
            
            # Save individual cleaned signal file
            np.save(os.path.join(clean_data_dir, f"{record_name}_cleaned.npy"), ecg_cleaned)
            
            # Extract features
            feature_dict = extractor.extract_durations(ecg_cleaned, fs)
            feature_dict["record"] = record_name
            feature_dict["fs"] = fs
            feature_results.append(feature_dict)
            
            # Check if we've reached a visualization checkpoint
            # Also trigger visualization if we've processed 50 more records since last checkpoint
            at_checkpoint = (next_checkpoint_idx < len(visualization_checkpoints) and 
                            record_count >= visualization_checkpoints[next_checkpoint_idx])
            
            if at_checkpoint or record_count % 50 == 0:
                # Update cumulative statistics and create visualizations
                update_cumulative_stats(cumulative_stats, feature_results, record_count)
                
                # Save intermediate features to CSV
                df_features = pd.DataFrame(feature_results)
                df_features.to_csv(
                    features_file,
                    index=False,
                    mode='w'  # Overwrite with complete dataset
                )
                
                # Create/update visualizations at checkpoints
                if at_checkpoint:
                    print(f"[INFO] Reached checkpoint: {record_count} records processed")
                    create_cumulative_visualizations(cumulative_stats, cumulative_dir)
                    next_checkpoint_idx += 1
                
                print(f"[INFO] Processed {record_count} records. Cumulative statistics updated.")
        
        except Exception as e:
            print(f"[ERROR] Processing record {record_name} failed: {e}")
            continue

    # Final update to stats and visualizations
    if feature_results:
        update_cumulative_stats(cumulative_stats, feature_results, record_count)
        
        # Save final features
        df_features = pd.DataFrame(feature_results)
        df_features.to_csv(features_file, index=False)
        
        # Create final visualizations
        create_cumulative_visualizations(cumulative_stats, cumulative_dir, is_final=True)
    
    print(f"\n[INFO] Finished. Processed {record_count} .hea/.dat pairs.")
    print(f"[INFO] Feature results saved to {features_file}")
    print(f"[INFO] Individual cleaned data files saved to {clean_data_dir}")
    print(f"[INFO] Cumulative visualizations saved to {cumulative_dir}")
    
    # Create additional visualizations using the Plots class
    if feature_results:
        print("\n[INFO] Creating additional visualizations using the Plots class...")
        create_final_visualizations(df_features, visualization_dir)


def update_cumulative_stats(stats, feature_results, record_count):
    """
    Update cumulative statistics dictionary with current averages.
    
    Parameters:
    -----------
    stats : dict
        Dictionary to store the evolving statistics
    feature_results : list
        List of dictionaries containing extracted features
    record_count : int
        Current count of processed records
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(feature_results)
    
    # Add the current record count
    stats['record_counts'].append(record_count)
    stats['processing_batches'].append(len(stats['record_counts']))
    
    # Calculate and store current averages for each feature
    for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
        if feature in df.columns:
            # Calculate the mean, ignoring NaN values
            current_avg = df[feature].dropna().mean()
            stats[f'{feature}_avg'].append(current_avg)
        else:
            # Feature not available, add NaN to maintain alignment
            stats[f'{feature}_avg'].append(np.nan)


def create_cumulative_visualizations(stats, output_dir, is_final=False):
    """
    Create visualizations showing how statistics evolve as more records are processed.
    
    Parameters:
    -----------
    stats : dict
        Dictionary with cumulative statistics
    output_dir : str
        Directory to save visualizations
    is_final : bool
        Whether this is the final visualization after all processing is complete
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a plot showing the evolution of feature averages
    plt.figure(figsize=(12, 8))
    
    # Plot each feature's average over time
    for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
        feature_key = f'{feature}_avg'
        if feature_key in stats and len(stats[feature_key]) > 0:
            plt.plot(stats['record_counts'], stats[feature_key], marker='o', label=feature)
    
    plt.xscale('log')  # Log scale for record count to better show early changes
    plt.xlabel('Number of Records Processed')
    plt.ylabel('Average Value (ms)')
    plt.title('Evolution of ECG Feature Averages During Processing')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add vertical lines at processing batch points
    for batch, count in zip(stats['processing_batches'], stats['record_counts']):
        plt.axvline(x=count, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    filename = 'feature_evolution_final.png' if is_final else 'feature_evolution_current.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Create a table of current values
    if len(stats['record_counts']) > 0:
        latest_idx = len(stats['record_counts']) - 1
        latest_count = stats['record_counts'][latest_idx]
        
        # Create a summary table
        plt.figure(figsize=(8, 6))
        plt.axis('tight')
        plt.axis('off')
        
        # Prepare table data
        table_data = []
        header = ['Feature', f'Current Avg (after {latest_count} records)']
        
        for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
            feature_key = f'{feature}_avg'
            if feature_key in stats and len(stats[feature_key]) > 0:
                current_value = stats[feature_key][latest_idx]
                if not np.isnan(current_value):
                    table_data.append([feature, f"{current_value:.2f} ms"])
                else:
                    table_data.append([feature, "N/A"])
        
        # Create and display the table
        plt.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
        plt.title(f'Current Feature Averages (after {latest_count} records)')
        plt.tight_layout()
        
        # Save the table
        filename = 'current_averages_final.png' if is_final else 'current_averages.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    # Additional visualization if there's enough data and this is the final visualization
    if is_final and len(stats['record_counts']) > 3:
        # Create a box plot of the final distribution of each feature
        plt.figure(figsize=(10, 7))
        
        # Last point data for box plot
        boxes_data = []
        labels = []
        
        for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
            feature_key = f'{feature}_avg'
            if feature_key in stats and len(stats[feature_key]) > 0:
                # Filter out NaN values
                feature_values = [v for v in stats[feature_key] if not np.isnan(v)]
                if feature_values:
                    boxes_data.append(feature_values)
                    labels.append(feature)
        
        if boxes_data:
            plt.boxplot(boxes_data, labels=labels)
            plt.title('Distribution of Feature Averages Across Processing')
            plt.ylabel('Value (ms)')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_distribution_across_processing.png'))
            plt.close()


def create_final_visualizations(df_features, visualization_dir):
    """
    Create additional visualizations using the Plots class.
    
    Parameters:
    -----------
    df_features : pandas.DataFrame
        DataFrame containing feature data
    visualization_dir : str
        Directory to save visualizations
    """
    # Make sure we have data to visualize
    if df_features.empty:
        print("[WARNING] No feature data available for visualization.")
        return
    
    # Create subdirectory for feature plots
    feature_plots_dir = os.path.join(visualization_dir, "feature_plots")
    os.makedirs(feature_plots_dir, exist_ok=True)
    
    # Original working directory
    original_dir = os.getcwd()
    os.chdir(feature_plots_dir)
    
    try:
        # Add dummy columns needed for specific plot types
        if 'arrhythmia' not in df_features.columns and 'QRS_duration' in df_features.columns:
            # Mark records with QRS > 120ms as possible arrhythmia
            df_features['arrhythmia'] = df_features['QRS_duration'] > 120
        
        if 'abnormal_hr' not in df_features.columns and 'QT_duration' in df_features.columns:
            # Mark records with QT > 450ms as abnormal heart rate
            df_features['abnormal_hr'] = df_features['QT_duration'] > 450
        
        # Initialize Plots class with feature data
        plots = Plots(labeled_data=df_features)
        
        # Generate feature distribution plots
        for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
            if feature in df_features.columns:
                try:
                    plots.plot_feature_distribution(feature=feature, save=True)
                    print(f"  - Created distribution plot for {feature}")
                except Exception as e:
                    print(f"  - Failed to create distribution plot for {feature}: {e}")
        
        # Generate missing data visualizations
        try:
            plots.plot_missing_bar(save=True)
            plots.plot_missing_heatmap(save=True)
            print("  - Created missing data visualizations")
        except Exception as e:
            print(f"  - Failed to create missing data visualizations: {e}")
        
        # Generate comparison plots if applicable
        if 'arrhythmia' in df_features.columns:
            for feature in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
                if feature in df_features.columns:
                    try:
                        plots.plot_arrhythmia_comparison(feature=feature, save=True)
                        print(f"  - Created arrhythmia comparison plot for {feature}")
                    except Exception as e:
                        print(f"  - Failed to create arrhythmia comparison: {e}")
        
        if 'abnormal_hr' in df_features.columns and 'QT_duration' in df_features.columns:
            try:
                plots.plot_abnormal_heart_rate_comparison(hr_column='QT_duration', save=True)
                print("  - Created abnormal heart rate comparison")
            except Exception as e:
                print(f"  - Failed to create heart rate comparison: {e}")
        
        print(f"[INFO] Additional visualizations saved to {feature_plots_dir}")
    
    except Exception as e:
        print(f"[ERROR] Visualization process failed: {e}")
    
    finally:
        # Restore original working directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()