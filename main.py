import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from pathlib import Path
from data_processing.cleaner import ECGCleaner
from data_processing.loader import ECGLoader
from feature_extraction.extractor import ECGFeatureExtractor

def compare_measurements(features_csv, machine_csv, output_csv=None):
    """
    Compare extracted ECG durations with machine measurements.
    Handles NULL values (29999) in machine measurements.

    Parameters
    ----------
    features_csv : str
        Path to CSV file with extracted ECG features.
    machine_csv : str
        Path to CSV file with machine measurement values.
    output_csv : str, optional
        If provided, saves the merged comparison result to this CSV.
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe with computed differences.
    """
    import pandas as pd
    import numpy as np
    
    # Load both CSV files
    features_df = pd.read_csv(features_csv)
    machine_df = pd.read_csv(machine_csv, low_memory=False)
    
    print(f"[INFO] Loaded features: {features_df.shape[0]} records")
    print(f"[INFO] Loaded machine measurements: {machine_df.shape[0]} records")
    
    # Remove 'p' prefix from patient_id and 's' prefix from study_id
    features_df['patient_id_numeric'] = features_df['patient_id'].str.replace('p', '', regex=False)
    features_df['study_id_numeric'] = features_df['study_id'].str.replace('s', '', regex=False)
    
    print(f"[INFO] Original patient_id example: {features_df['patient_id'].iloc[0]}")
    print(f"[INFO] Modified patient_id example: {features_df['patient_id_numeric'].iloc[0]}")
    print(f"[INFO] Original study_id example: {features_df['study_id'].iloc[0]}")
    print(f"[INFO] Modified study_id example: {features_df['study_id_numeric'].iloc[0]}")
    print(f"[INFO] Subject_id example from machine: {machine_df['subject_id'].iloc[0]}")
    print(f"[INFO] Study_id example from machine: {machine_df['study_id'].iloc[0]}")
    
    # Convert all ID columns to strings
    features_df['patient_id_numeric'] = features_df['patient_id_numeric'].astype(str)
    features_df['study_id_numeric'] = features_df['study_id_numeric'].astype(str)
    machine_df['subject_id'] = machine_df['subject_id'].astype(str)
    machine_df['study_id'] = machine_df['study_id'].astype(str)
    
    # Print column information for debugging
    print(f"[INFO] Features DataFrame columns: {features_df.columns.tolist()}")
    print(f"[INFO] Machine DataFrame columns: {machine_df.columns.tolist()}")
    
    # Merge dataframes on the modified keys
    print("[INFO] Merging datasets on patient_id_numeric/subject_id and study_id_numeric/study_id...")
    merged = pd.merge(
        features_df, machine_df,
        left_on=["patient_id_numeric", "study_id_numeric"],
        right_on=["subject_id", "study_id"],
        how="inner"
    )
    
    print(f"[INFO] Matched records after merge: {merged.shape[0]}")

    # If no matches, provide diagnostic info
    if merged.shape[0] == 0:
        print("[WARNING] No matching records found! Checking for potential issues...")
        # Sample some values for comparison
        print(f"[INFO] First 5 patient_id_numeric from features: {features_df['patient_id_numeric'].head().tolist()}")
        print(f"[INFO] First 5 subject_id from machine: {machine_df['subject_id'].head().tolist()}")
        print(f"[INFO] First 5 study_id_numeric from features: {features_df['study_id_numeric'].head().tolist()}")
        print(f"[INFO] First 5 study_id from machine: {machine_df['study_id'].head().tolist()}")
        
        # Try a direct match on record_id and study_id_numeric to help diagnose
        print("[INFO] Attempting to find matches between record_id and machine study_id...")
        for i, record_id in enumerate(features_df['record_id'].head(10)):
            matches = machine_df[machine_df['study_id'] == record_id]
            if not matches.empty:
                print(f"[INFO] Found match! record_id {record_id} matches machine study_id")
                print(f"[INFO] Feature patient_id: {features_df.iloc[i]['patient_id']}, machine subject_id: {matches.iloc[0]['subject_id']}")
        
        return pd.DataFrame()  # Return empty dataframe

    # Handle NULL values in machine measurements (29999)
    # This is treating 29999 as NULL values, which is correct
    NULL_VALUE = 29999
    
    # Replace NULL values with NaN
    print("[INFO] Replacing NULL values (29999) in machine measurements with NaN...")
    columns_to_check = ['p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
    for col in columns_to_check:
        if col in merged.columns:
            null_count = (merged[col] == NULL_VALUE).sum()
            if null_count > 0:
                print(f"[INFO] Found {null_count} NULL values in {col}")
                merged[col] = merged[col].replace(NULL_VALUE, np.nan)
    
    # Compute machine-derived durations (in milliseconds), skipping rows with NULL values
    print("[INFO] Computing durations and differences (skipping NULL values)...")
    
    # Create mask for valid QRS values
    valid_qrs_mask = merged['qrs_onset'].notna() & merged['qrs_end'].notna()
    print(f"[INFO] Valid QRS measurements: {valid_qrs_mask.sum()} out of {len(merged)}")
    
    # Create mask for valid P values
    valid_p_mask = merged['p_onset'].notna() & merged['p_end'].notna()
    print(f"[INFO] Valid P wave measurements: {valid_p_mask.sum()} out of {len(merged)}")
    
    # Create mask for valid PQ values
    valid_pq_mask = merged['p_onset'].notna() & merged['qrs_onset'].notna()
    print(f"[INFO] Valid PQ interval measurements: {valid_pq_mask.sum()} out of {len(merged)}")
    
    # Create mask for valid QT values
    valid_qt_mask = merged['qrs_onset'].notna() & merged['t_end'].notna()
    print(f"[INFO] Valid QT interval measurements: {valid_qt_mask.sum()} out of {len(merged)}")
    
    # Initialize columns with NaN
    merged["Machine_QRS_Duration"] = np.nan
    merged["Machine_P_Duration"] = np.nan
    merged["Machine_PQ_Duration"] = np.nan
    merged["Machine_QT_Duration"] = np.nan
    
    # Only calculate durations for valid measurements
    merged.loc[valid_qrs_mask, "Machine_QRS_Duration"] = merged.loc[valid_qrs_mask, "qrs_end"] - merged.loc[valid_qrs_mask, "qrs_onset"]
    merged.loc[valid_p_mask, "Machine_P_Duration"] = merged.loc[valid_p_mask, "p_end"] - merged.loc[valid_p_mask, "p_onset"]
    merged.loc[valid_pq_mask, "Machine_PQ_Duration"] = merged.loc[valid_pq_mask, "qrs_onset"] - merged.loc[valid_pq_mask, "p_onset"]
    merged.loc[valid_qt_mask, "Machine_QT_Duration"] = merged.loc[valid_qt_mask, "t_end"] - merged.loc[valid_qt_mask, "qrs_onset"]
    
    # Compute differences between extracted and machine durations (only for valid measurements)
    merged["Diff_QRS"] = merged["QRS_Duration_Mean"] - merged["Machine_QRS_Duration"]
    merged["Diff_P"] = merged["P_Duration_Mean"] - merged["Machine_P_Duration"]
    merged["Diff_PQ"] = merged["PQ_Interval_Mean"] - merged["Machine_PQ_Duration"]
    merged["Diff_QT"] = merged["QT_Interval_Mean"] - merged["Machine_QT_Duration"]

    # Compute absolute differences for evaluation
    merged["AbsDiff_QRS"] = merged["Diff_QRS"].abs()
    merged["AbsDiff_P"] = merged["Diff_P"].abs()
    merged["AbsDiff_PQ"] = merged["Diff_PQ"].abs()
    merged["AbsDiff_QT"] = merged["Diff_QT"].abs()

    # Print summary statistics for the absolute differences (excluding NaN values)
    summary = merged[["AbsDiff_QRS", "AbsDiff_P", "AbsDiff_PQ", "AbsDiff_QT"]].describe()
    print("\n=== Comparison Summary ===")
    print(summary)

    # Optionally, save the merged result with comparisons to a CSV file
    if output_csv:
        print(f"[INFO] Saving comparison results to: {output_csv}")
        merged.to_csv(output_csv, index=False)
        print(f"[INFO] Comparison results saved to: {output_csv}")
        
    # Print some specific examples
    if not merged.empty:
        print("\n[INFO] Sample comparisons (first 5 records):")
        sample_cols = ['record_id', 'QRS_Duration_Mean', 'Machine_QRS_Duration', 'Diff_QRS', 
                       'QT_Interval_Mean', 'Machine_QT_Duration', 'Diff_QT']
        print(merged[sample_cols].head(5))
        
        # Print average differences to help with calibration (excluding NaN values)
        print("\n[INFO] Average differences:")
        print(f"QRS Duration: {merged['Diff_QRS'].mean():.2f} ms")
        print(f"P Duration: {merged['Diff_P'].mean():.2f} ms")
        print(f"PQ Interval: {merged['Diff_PQ'].mean():.2f} ms")
        print(f"QT Interval: {merged['Diff_QT'].mean():.2f} ms")
        
        # Print recommended correction factors based on the averages (for non-NaN values)
        valid_qrs = merged.dropna(subset=['Machine_QRS_Duration', 'QRS_Duration_Mean'])
        if not valid_qrs.empty and abs(valid_qrs['Diff_QRS'].mean()) > 5:
            qrs_factor = valid_qrs['Machine_QRS_Duration'].mean() / valid_qrs['QRS_Duration_Mean'].mean()
            print(f"[INFO] Recommended QRS correction factor: {qrs_factor:.3f}")
        
        valid_qt = merged.dropna(subset=['Machine_QT_Duration', 'QT_Interval_Mean'])
        if not valid_qt.empty and abs(valid_qt['Diff_QT'].mean()) > 5:
            qt_factor = valid_qt['Machine_QT_Duration'].mean() / valid_qt['QT_Interval_Mean'].mean()
            print(f"[INFO] Recommended QT correction factor: {qt_factor:.3f}")
        
        valid_p = merged.dropna(subset=['Machine_P_Duration', 'P_Duration_Mean'])
        if not valid_p.empty and abs(valid_p['Diff_P'].mean()) > 5:
            p_factor = valid_p['Machine_P_Duration'].mean() / valid_p['P_Duration_Mean'].mean()
            print(f"[INFO] Recommended P correction factor: {p_factor:.3f}")
        
        valid_pq = merged.dropna(subset=['Machine_PQ_Duration', 'PQ_Interval_Mean'])
        if not valid_pq.empty and abs(valid_pq['Diff_PQ'].mean()) > 5:
            pq_factor = valid_pq['Machine_PQ_Duration'].mean() / valid_pq['PQ_Interval_Mean'].mean()
            print(f"[INFO] Recommended PQ correction factor: {pq_factor:.3f}")

    return merged

def run_comparison_only(features_csv, machine_csv, output_csv):
    """
    Run just the comparison part of the pipeline - useful after
    stopping the main processing with Ctrl+C.
    
    Parameters
    ----------
    features_csv : str
        Path to CSV file with extracted ECG features.
    machine_csv : str
        Path to CSV file with machine measurement values.
    output_csv : str
        Path to save comparison results to.
    """
    print("\n[INFO] Running comparison only mode...")
    
    # Make sure paths exist
    if not os.path.exists(features_csv):
        print(f"[ERROR] Features CSV not found at: {features_csv}")
        return
        
    if not os.path.exists(machine_csv):
        print(f"[ERROR] Machine measurements CSV not found at: {machine_csv}")
        return
    
    print(f"[INFO] Features CSV: {features_csv}")
    print(f"[INFO] Machine CSV: {machine_csv}")
    print(f"[INFO] Output will be saved to: {output_csv}")
    
    # Run comparison
    print("[INFO] Comparing extracted features with machine measurements...")
    comparison_df = compare_measurements(features_csv, machine_csv, output_csv)
    
    if not comparison_df.empty:
        print("[INFO] Comparison complete!")
    else:
        print("[WARNING] Comparison completed but no matching records were found.")
        print("[INFO] Check that your ID formats match between the two files.")
        print("\n[INFO] Possible solutions:")
        print("1. Try matching on record_id directly (it might be the same as machine study_id)")
        print("2. Check if your folder structure in the data matches the naming convention expected")
        print("3. Consider manually creating a mapping file between your IDs and machine IDs")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Load, clean, extract features from ECG data, and compare with machine measurements'
    )
    parser.add_argument('--input_dir', type=str, 
                        default='data/external/cleaned-dataset/relevant_data/files',
                        help='Directory containing the .hea/.dat files')
    parser.add_argument('--output_dir', type=str, 
                        default='data/external/cleaned_ecg',
                        help='Directory to save cleaned data')
    parser.add_argument('--labels_csv', type=str, 
                        default=None,
                        help='Path to CSV file with labels (optional)')
    parser.add_argument('--sampling_rate', type=float, 
                        default=500.0,
                        help='ECG sampling rate in Hz')
    parser.add_argument('--start_from', type=int,
                        default=0,
                        help='Start processing from this record number (0-indexed)')
    parser.add_argument('--features_csv', type=str,
                        default='ecg_features.csv',
                        help='Filename for extracted features CSV')
    parser.add_argument('--machine_csv', type=str, default=None,
                        help='Path to CSV file with machine measurements for comparison (optional)')
    parser.add_argument('--comparison_only', action='store_true',
                        help='Skip processing and only run comparison (use after Ctrl+C)')
    args = parser.parse_args()
    
    # Create absolute paths
    base_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define features CSV path
    features_csv_path = os.path.join(output_dir, args.features_csv)
    
    # Set comparison output path
    comparison_output_csv = os.path.join(output_dir, "comparison_results.csv")
    
    # If running in comparison-only mode, just do the comparison and exit
    if args.comparison_only:
        if not args.machine_csv:
            print("[ERROR] --machine_csv must be provided with --comparison_only")
            return
        run_comparison_only(features_csv_path, args.machine_csv, comparison_output_csv)
        return
    
    # Normal processing mode - continue with ECG processing
    # Ask user in terminal how many records to skip
    skip_input = input("How many records would you like to skip? [0]: ")
    skip_count = int(skip_input) if skip_input.strip() else 0
    print(f"[INFO] Will skip {skip_count} records")
    
    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Cleaned data will be saved to: {output_dir} (preserving original folder structure)")
    print(f"[INFO] ECG features will be saved to: {features_csv_path}")
    
    # Initialize loader, cleaner, and feature extractor
    loader = ECGLoader(base_dir, args.labels_csv)
    cleaner = ECGCleaner(fs=args.sampling_rate)
    extractor = ECGFeatureExtractor(sampling_rate=args.sampling_rate)
    
    print(f"[DEBUG] ECGLoader initialized with base_dir: {base_dir}")
    print(f"[DEBUG] ECGCleaner initialized with sampling rate: {args.sampling_rate} Hz")
    print(f"[DEBUG] ECGFeatureExtractor initialized with sampling rate: {args.sampling_rate} Hz")
    
    # If labels CSV is provided, load it
    if args.labels_csv:
        labels_df = loader.load_csv_labels()
    
    # Process each ECG record
    found_records = False
    count = 0
    processed = 0
    errors = 0
    
    # Check if features CSV exists and create with headers if it doesn't
    features_file_exists = os.path.isfile(features_csv_path)
    
    print("[DEBUG] Starting to iterate through ECG records...")
    skipped = 0
    for hea_path, dat_path in loader.iter_ecg_records():
        found_records = True
        
        # Skip records if requested
        if skipped < skip_count:
            skipped += 1
            continue
            
        count += 1
        
        # Skip records below start_from
        if count < args.start_from + 1:
            continue
        
        try:
            # Extract record base (path without extension)
            record_base = os.path.splitext(hea_path)[0]
            record_id = os.path.basename(record_base)
            
            # Create relative path to maintain folder structure
            rel_path = os.path.relpath(os.path.dirname(hea_path), base_dir)
            target_dir = os.path.join(output_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"[INFO] Processing record {count}: {record_id}")
            print(f"[INFO] HEA path: {hea_path}")
            print(f"[INFO] DAT path: {dat_path}")
            
            # Load the ECG data
            signals, fields = loader.load_ecg_record(record_base)
            print(f"[INFO] Signal shape before cleaning: {signals.shape}")
            
            # Clean the ECG data
            cleaned_signals = cleaner.clean_ecg(signals)
            print(f"[INFO] Signal shape after cleaning: {cleaned_signals.shape}")
            
            # Prepare output paths - preserve original filename but add -cleaned suffix
            output_record_name = f"{record_id}-cleaned"
            output_record_path = os.path.join(target_dir, output_record_name)
            
            # Save as WFDB format with correct record name and write_dir parameters
            wfdb.wrsamp(
                record_name=output_record_name,
                fs=fields['fs'],
                units=fields['units'],
                sig_name=fields['sig_name'],
                p_signal=cleaned_signals,
                fmt=['16'] * cleaned_signals.shape[1],
                write_dir=target_dir
            )
            print(f"[INFO] Saved cleaned data to: {output_record_path} in WFDB format")
            
            # Extract features from cleaned ECG
            print(f"[INFO] Extracting features from cleaned ECG")
            features = extractor.extract_features(
                cleaned_signals[:, 1] if cleaned_signals.shape[1] >= 2 else cleaned_signals[:, 0]
            )
            
            # Add metadata to features
            features['record_id'] = record_id
            features['patient_id'] = os.path.basename(os.path.dirname(os.path.dirname(hea_path)))
            features['study_id'] = os.path.basename(os.path.dirname(hea_path))
            features['original_path'] = record_base
            features['cleaned_path'] = output_record_path
            
            # Save features to CSV (append mode)
            features_df = pd.DataFrame([features])
            mode = 'w' if not features_file_exists else 'a'
            header = not features_file_exists
            features_df.to_csv(features_csv_path, mode=mode, header=header, index=False)
            
            # Update flag after first write
            if not features_file_exists:
                features_file_exists = True
                
            print(f"[INFO] Appended features to: {features_csv_path}")
            processed += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process record {hea_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    if not found_records:
        print("[WARNING] No valid ECG records found! Check the input directory structure.")
        if not os.path.exists(base_dir):
            print(f"[ERROR] Input directory does not exist: {base_dir}")
        else:
            print(f"[DEBUG] Files in input directory:")
            for root, dirs, files in os.walk(base_dir):
                print(f"  Directory: {root}")
                for file in files:
                    print(f"    - {file}")
    elif processed == 0:
        print(f"[WARNING] Skipped all {skipped} records found. Check your skip count and start_from parameters.")
    
    print("\n[SUMMARY] Finished ECG processing:")
    print(f"  - Processed {processed} records")
    print(f"  - Skipped {skipped} records")
    print(f"  - Encountered {errors} errors")
    print(f"  - Cleaned data saved to {output_dir} (preserving original folder structure)")
    print(f"  - ECG features saved to {features_csv_path}")

    # Compare extracted features with machine measurements if provided
    if args.machine_csv:
        print("[INFO] Comparing extracted features with machine measurements...")
        comparison_df = compare_measurements(features_csv_path, args.machine_csv, comparison_output_csv)
        if not comparison_df.empty:
            print(f"[INFO] Comparison results saved to: {comparison_output_csv}")
        else:
            print("[WARNING] Comparison completed but no matching records were found.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user (Ctrl+C)")
        print("[INFO] You can run the comparison part separately with:")
        print("       python main.py --comparison_only --machine_csv \"path/to/machine_measurements.csv\"")