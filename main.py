import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from pathlib import Path
# Assuming these modules exist in your project structure
from data_processing.cleaner import ECGCleaner
from data_processing.loader import ECGLoader
from feature_extraction.extractor import ECGFeatureExtractor

# =============================================================================
# Comparison Function (with Fix)
# =============================================================================
def compare_measurements(features_csv, machine_csv, output_csv=None):
    """
    Compare extracted ECG durations with machine measurements.
    Handles NULL values (29999) and ensures numeric types before calculation.

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
    try:
        features_df = pd.read_csv(features_csv)
        print(f"[INFO] Loaded features: {features_df.shape[0]} records from {features_csv}")
    except FileNotFoundError:
        print(f"[ERROR] Features CSV not found at: {features_csv}")
        return pd.DataFrame() # Return empty dataframe if features file not found
    except Exception as e:
        print(f"[ERROR] Failed to load features CSV {features_csv}: {e}")
        return pd.DataFrame()

    try:
        # Explicitly define dtypes for potentially problematic columns if needed,
        # but pd.to_numeric later should handle most cases.
        machine_df = pd.read_csv(machine_csv, low_memory=False)
        print(f"[INFO] Loaded machine measurements: {machine_df.shape[0]} records from {machine_csv}")
    except FileNotFoundError:
        print(f"[ERROR] Machine measurements CSV not found at: {machine_csv}")
        return pd.DataFrame() # Return empty dataframe if machine file not found
    except Exception as e:
        print(f"[ERROR] Failed to load machine measurements CSV {machine_csv}: {e}")
        return pd.DataFrame()

    # --- Data Preparation and Merging ---
    # Remove 'p' prefix from patient_id and 's' prefix from study_id
    if 'patient_id' in features_df.columns and 'study_id' in features_df.columns:
        features_df['patient_id_numeric'] = features_df['patient_id'].astype(str).str.replace('p', '', regex=False)
        features_df['study_id_numeric'] = features_df['study_id'].astype(str).str.replace('s', '', regex=False)
        print(f"[INFO] Original patient_id example: {features_df['patient_id'].iloc[0] if not features_df.empty else 'N/A'}")
        print(f"[INFO] Modified patient_id example: {features_df['patient_id_numeric'].iloc[0] if not features_df.empty else 'N/A'}")
        print(f"[INFO] Original study_id example: {features_df['study_id'].iloc[0] if not features_df.empty else 'N/A'}")
        print(f"[INFO] Modified study_id example: {features_df['study_id_numeric'].iloc[0] if not features_df.empty else 'N/A'}")
    else:
        print("[ERROR] 'patient_id' or 'study_id' columns not found in features_df.")
        return pd.DataFrame()

    if 'subject_id' in machine_df.columns and 'study_id' in machine_df.columns:
        print(f"[INFO] Subject_id example from machine: {machine_df['subject_id'].iloc[0] if not machine_df.empty else 'N/A'}")
        print(f"[INFO] Study_id example from machine: {machine_df['study_id'].iloc[0] if not machine_df.empty else 'N/A'}")
        # Convert all ID columns to strings for reliable merging
        machine_df['subject_id'] = machine_df['subject_id'].astype(str)
        machine_df['study_id'] = machine_df['study_id'].astype(str)
    else:
        print("[ERROR] 'subject_id' or 'study_id' columns not found in machine_df.")
        return pd.DataFrame()

    # Ensure feature IDs are also strings
    features_df['patient_id_numeric'] = features_df['patient_id_numeric'].astype(str)
    features_df['study_id_numeric'] = features_df['study_id_numeric'].astype(str)

    # Print column information for debugging
    print(f"[INFO] Features DataFrame columns: {features_df.columns.tolist()}")
    print(f"[INFO] Machine DataFrame columns: {machine_df.columns.tolist()}")

    # Merge dataframes on the modified keys
    print("[INFO] Merging datasets on patient_id_numeric/subject_id and study_id_numeric/study_id...")
    merged = pd.merge(
        features_df, machine_df,
        left_on=["patient_id_numeric", "study_id_numeric"],
        right_on=["subject_id", "study_id"],
        how="inner" # Use 'inner' to keep only matching records
    )

    print(f"[INFO] Matched records after merge: {merged.shape[0]}")

    # If no matches, provide diagnostic info and return
    if merged.shape[0] == 0:
        print("[WARNING] No matching records found after merge! Check ID formats and values.")
        # (Optional: Add more diagnostic prints here if needed)
        return pd.DataFrame() # Return empty dataframe

    # --- Explicit Numeric Conversion (THE FIX) ---
    NULL_VALUE = 29999
    # Columns from machine_df needed for calculations
    machine_calc_cols = ['p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end']
    # Columns from features_df needed for calculations
    feature_calc_cols = ["QRS_Duration_Mean", "P_Duration_Mean", "PQ_Interval_Mean", "QT_Interval_Mean"]
    # Other machine columns to convert (optional, but good practice if used later)
    other_machine_cols_to_convert = ['p_axis', 'qrs_axis', 't_axis', 'rr_interval']

    all_cols_to_convert = machine_calc_cols + feature_calc_cols + other_machine_cols_to_convert

    print("[INFO] Converting relevant columns to numeric and handling NULLs...")
    for col in all_cols_to_convert:
        if col in merged.columns:
            original_dtype = merged[col].dtype
            nan_before = merged[col].isna().sum()

            # Attempt to replace the specific NULL value first (if column isn't already float)
            # This handles cases where 29999 might be read as integer or string
            if not pd.api.types.is_float_dtype(merged[col]):
                 try:
                     # Use .loc to avoid SettingWithCopyWarning if applicable
                     merged.loc[merged[col] == NULL_VALUE, col] = np.nan
                 except TypeError:
                      # If direct comparison fails (e.g., mixed types), convert to string first
                      try:
                           is_null_str = merged[col].astype(str) == str(NULL_VALUE)
                           merged.loc[is_null_str, col] = np.nan
                      except Exception as e:
                           print(f"[WARNING] Could not replace NULL {NULL_VALUE} in column {col} before numeric conversion: {e}")

            # Convert the entire column to numeric, coercing errors
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

            nan_after = merged[col].isna().sum()
            print(f"  - Column '{col}': {original_dtype} -> {merged[col].dtype}. NaNs: {nan_before} -> {nan_after}.")
        else:
            print(f"  - Column '{col}': Not found in merged dataframe.")


    # --- Duration Calculation (Now safe with numeric columns) ---
    print("[INFO] Computing machine durations and differences...")

    # Create masks (relies on NaNs from previous step)
    valid_qrs_mask = merged['qrs_onset'].notna() & merged['qrs_end'].notna()
    valid_p_mask = merged['p_onset'].notna() & merged['p_end'].notna()
    valid_pq_mask = merged['p_onset'].notna() & merged['qrs_onset'].notna()
    valid_qt_mask = merged['qrs_onset'].notna() & merged['t_end'].notna()

    print(f"[INFO] Valid records for QRS duration calculation: {valid_qrs_mask.sum()}")
    print(f"[INFO] Valid records for P duration calculation: {valid_p_mask.sum()}")
    print(f"[INFO] Valid records for PQ interval calculation: {valid_pq_mask.sum()}")
    print(f"[INFO] Valid records for QT interval calculation: {valid_qt_mask.sum()}")

    # Initialize columns with NaN
    merged["Machine_QRS_Duration"] = np.nan
    merged["Machine_P_Duration"] = np.nan
    merged["Machine_PQ_Duration"] = np.nan
    merged["Machine_QT_Duration"] = np.nan

    # Calculate durations only for valid measurements using .loc
    merged.loc[valid_qrs_mask, "Machine_QRS_Duration"] = merged.loc[valid_qrs_mask, "qrs_end"] - merged.loc[valid_qrs_mask, "qrs_onset"]
    merged.loc[valid_p_mask, "Machine_P_Duration"] = merged.loc[valid_p_mask, "p_end"] - merged.loc[valid_p_mask, "p_onset"]
    merged.loc[valid_pq_mask, "Machine_PQ_Duration"] = merged.loc[valid_pq_mask, "qrs_onset"] - merged.loc[valid_pq_mask, "p_onset"]
    merged.loc[valid_qt_mask, "Machine_QT_Duration"] = merged.loc[valid_qt_mask, "t_end"] - merged.loc[valid_qt_mask, "qrs_onset"]

    # Compute differences between extracted and machine durations
    # These operations will now work correctly (result is NaN if any operand is NaN)
    merged["Diff_QRS"] = merged["QRS_Duration_Mean"] - merged["Machine_QRS_Duration"]
    merged["Diff_P"] = merged["P_Duration_Mean"] - merged["Machine_P_Duration"]
    merged["Diff_PQ"] = merged["PQ_Interval_Mean"] - merged["Machine_PQ_Duration"]
    merged["Diff_QT"] = merged["QT_Interval_Mean"] - merged["Machine_QT_Duration"]

    # Compute absolute differences for evaluation
    merged["AbsDiff_QRS"] = merged["Diff_QRS"].abs()
    merged["AbsDiff_P"] = merged["Diff_P"].abs()
    merged["AbsDiff_PQ"] = merged["Diff_PQ"].abs()
    merged["AbsDiff_QT"] = merged["Diff_QT"].abs()

    # --- Results and Output ---
    # Print summary statistics for the absolute differences (excluding NaN values)
    summary = merged[["AbsDiff_QRS", "AbsDiff_P", "AbsDiff_PQ", "AbsDiff_QT"]].describe()
    print("\n=== Comparison Summary (Absolute Differences) ===")
    print(summary)

    # Optionally, save the merged result with comparisons to a CSV file
    if output_csv:
        try:
            print(f"\n[INFO] Saving comparison results to: {output_csv}")
            merged.to_csv(output_csv, index=False, float_format='%.3f') # Format floats for readability
            print(f"[INFO] Comparison results saved.")
        except Exception as e:
            print(f"[ERROR] Failed to save comparison results to {output_csv}: {e}")

    # Print some specific examples
    if not merged.empty:
        print("\n[INFO] Sample comparisons (first 5 records):")
        sample_cols = ['record_id', 'QRS_Duration_Mean', 'Machine_QRS_Duration', 'Diff_QRS',
                       'QT_Interval_Mean', 'Machine_QT_Duration', 'Diff_QT']
        # Ensure columns exist before trying to print them
        cols_to_print = [col for col in sample_cols if col in merged.columns]
        print(merged[cols_to_print].head(5).to_string(float_format='%.2f')) # Format floats

        # Print average differences to help with calibration (excluding NaN values)
        print("\n[INFO] Average differences (Calculated - Machine):")
        print(f"QRS Duration: {merged['Diff_QRS'].mean():.2f} ms (based on {merged['Diff_QRS'].notna().sum()} pairs)")
        print(f"P Duration:   {merged['Diff_P'].mean():.2f} ms (based on {merged['Diff_P'].notna().sum()} pairs)")
        print(f"PQ Interval:  {merged['Diff_PQ'].mean():.2f} ms (based on {merged['Diff_PQ'].notna().sum()} pairs)")
        print(f"QT Interval:  {merged['Diff_QT'].mean():.2f} ms (based on {merged['Diff_QT'].notna().sum()} pairs)")

        # (Optional: Correction factor calculation - keep if needed)
        # print("\n[INFO] Potential Correction Factors (Machine / Calculated):")
        # # ... (keep the factor calculation logic if desired) ...

    return merged

# =============================================================================
# Comparison Only Runner
# =============================================================================
def run_comparison_only(features_csv, machine_csv, output_csv):
    """
    Run just the comparison part of the pipeline.
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
    print("\n[INFO] Comparing extracted features with machine measurements...")
    comparison_df = compare_measurements(features_csv, machine_csv, output_csv)

    if comparison_df is None:
        print("[ERROR] Comparison function failed.")
    elif not comparison_df.empty:
        print("\n[INFO] Comparison complete!")
    else:
        # This case handles both errors during loading/merging and zero matches
        print("\n[WARNING] Comparison completed but resulted in an empty dataframe.")
        print("[INFO] This could be due to file loading errors, no matching IDs found during merge, or other issues.")
        print("[INFO] Please check the logs above for specific errors or warnings.")
        print("[INFO] Verify ID formats and values in both input files.")


# =============================================================================
# Main Processing Logic
# =============================================================================
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
                        help='Directory to save cleaned data and features')
    parser.add_argument('--labels_csv', type=str,
                        default=None,
                        help='Path to CSV file with labels (optional)')
    parser.add_argument('--sampling_rate', type=float,
                        default=500.0,
                        help='ECG sampling rate in Hz')
    parser.add_argument('--start_from', type=int,
                        default=0,
                        help='Start processing from this record number (0-indexed, relative to found files)')
    parser.add_argument('--features_csv', type=str,
                        default='ecg_features.csv',
                        help='Filename for extracted features CSV within output_dir')
    parser.add_argument('--machine_csv', type=str, default=None,
                        help='Path to CSV file with machine measurements for comparison (optional)')
    parser.add_argument('--comparison_only', action='store_true',
                        help='Skip processing and only run comparison (requires existing features_csv)')
    parser.add_argument('--skip_records', type=int,
                        default=0,
                        help='Number of initial records to skip during processing')

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

    # --- Mode Selection ---
    if args.comparison_only:
        if not args.machine_csv:
            print("[ERROR] --machine_csv must be provided with --comparison_only")
            return
        if not os.path.exists(features_csv_path):
             print(f"[ERROR] Features CSV for comparison not found at expected location: {features_csv_path}")
             print(f"[INFO] Please ensure the feature extraction has run successfully first, or provide the correct path via --output_dir and --features_csv.")
             return
        # Make machine_csv path absolute if it's relative
        machine_csv_path = os.path.abspath(args.machine_csv) if args.machine_csv else None
        run_comparison_only(features_csv_path, machine_csv_path, comparison_output_csv)
        return

    # --- Normal Processing Mode ---
    print("--- Starting ECG Processing ---")
    print(f"[INFO] Input directory: {base_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Cleaned data will be saved within: {output_dir} (preserving structure)")
    print(f"[INFO] ECG features will be saved to: {features_csv_path}")
    print(f"[INFO] Sampling rate: {args.sampling_rate} Hz")
    print(f"[INFO] Starting from record index: {args.start_from}")
    print(f"[INFO] Skipping initial records: {args.skip_records}")
    if args.machine_csv:
        print(f"[INFO] Machine measurements CSV for comparison: {args.machine_csv}")


    # Initialize components
    loader = ECGLoader(base_dir, args.labels_csv)
    cleaner = ECGCleaner(fs=args.sampling_rate)
    extractor = ECGFeatureExtractor(sampling_rate=args.sampling_rate)

    # Load labels if provided (optional)
    # labels_df = None # Initialize
    # if args.labels_csv:
    #     try:
    #         labels_df = loader.load_csv_labels()
    #         print(f"[INFO] Loaded labels from: {args.labels_csv}")
    #     except Exception as e:
    #         print(f"[WARNING] Could not load labels CSV: {e}")


    # Process each ECG record
    found_records = False
    total_files_found = 0
    processed_count = 0
    error_count = 0
    skipped_count = 0

    # Check if features CSV exists and handle header writing
    features_file_exists = os.path.isfile(features_csv_path)
    if features_file_exists and args.start_from == 0 and args.skip_records == 0:
        print(f"[WARNING] Features file {features_csv_path} already exists. Appending new data.")
        # Consider adding an option to overwrite if desired
    elif not features_file_exists:
         print(f"[INFO] Features file {features_csv_path} does not exist. Will create it.")


    print("\n[INFO] Starting record iteration...")
    try:
        # Iterate through records using the loader
        for record_index, (hea_path, dat_path) in enumerate(loader.iter_ecg_records()):
            found_records = True
            total_files_found += 1

            # Handle skipping initial records
            if record_index < args.skip_records:
                skipped_count += 1
                if skipped_count % 100 == 0: # Print progress periodically
                     print(f"[INFO] Skipping... ({skipped_count}/{args.skip_records})", end='\r')
                continue
            if skipped_count == args.skip_records and record_index == args.skip_records:
                 print(f"\n[INFO] Finished skipping {skipped_count} records.")


            # Handle starting from a specific index (relative to non-skipped files)
            current_processing_index = record_index - args.skip_records
            if current_processing_index < args.start_from:
                 continue # Silently skip until start_from index is reached


            # --- Record Processing Core ---
            record_base = os.path.splitext(hea_path)[0]
            record_id = os.path.basename(record_base)
            print(f"\n[INFO] Processing record #{current_processing_index + 1} (File #{total_files_found}): {record_id}")
            # print(f"  - Path: {record_base}")

            try:
                # Create relative path for saving cleaned data
                rel_path = os.path.relpath(os.path.dirname(hea_path), base_dir)
                target_dir = os.path.join(output_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                # Load ECG data
                signals, fields = loader.load_ecg_record(record_base)
                # print(f"  - Original signal shape: {signals.shape}")

                # Clean ECG data
                cleaned_signals = cleaner.clean_ecg(signals)
                # print(f"  - Cleaned signal shape: {cleaned_signals.shape}")

                # Save cleaned data
                output_record_name = f"{record_id}-cleaned"
                output_record_path = os.path.join(target_dir, output_record_name)
                wfdb.wrsamp(
                    record_name=output_record_name,
                    fs=fields['fs'],
                    units=fields['units'],
                    sig_name=fields['sig_name'],
                    p_signal=cleaned_signals,
                    fmt=['16'] * cleaned_signals.shape[1], # Assuming 16-bit format
                    write_dir=target_dir
                )
                # print(f"  - Saved cleaned data to: {output_record_path}.hea/.dat")

                # Extract features
                # Use lead II (index 1) if available, otherwise lead I (index 0)
                lead_index_to_use = 1 if cleaned_signals.shape[1] >= 2 else 0
                # print(f"  - Extracting features from lead index: {lead_index_to_use}")
                features = extractor.extract_features(cleaned_signals[:, lead_index_to_use])

                # Add metadata
                features['record_id'] = record_id
                features['patient_id'] = os.path.basename(os.path.dirname(os.path.dirname(hea_path))) # Assumes patient/study/record structure
                features['study_id'] = os.path.basename(os.path.dirname(hea_path))
                features['original_path'] = os.path.relpath(record_base, os.path.dirname(output_dir)) # Store relative path
                features['cleaned_path'] = os.path.relpath(output_record_path, os.path.dirname(output_dir)) # Store relative path

                # Save features (append)
                features_df_single = pd.DataFrame([features])
                mode = 'a' if features_file_exists else 'w' # Append if exists, write otherwise
                header = not features_file_exists # Write header only if file doesn't exist
                features_df_single.to_csv(features_csv_path, mode=mode, header=header, index=False, float_format='%.3f')

                # Update flag after first write
                if not features_file_exists:
                    features_file_exists = True
                # print(f"  - Appended features to: {features_csv_path}")

                processed_count += 1

            except Exception as e:
                print(f"[ERROR] Failed processing record {record_id}: {e}")
                import traceback
                print(traceback.format_exc()) # Print full traceback for error
                error_count += 1
            # --- End Record Processing Core ---

    except KeyboardInterrupt:
         print("\n[INFO] Processing interrupted by user (Ctrl+C).")
         # Fall through to summary and potential comparison


    # --- Processing Summary ---
    print("\n--- Processing Summary ---")
    if not found_records:
        print("[WARNING] No ECG records (.hea/.dat files) found in the specified input directory.")
        print(f"  - Searched in: {base_dir}")
    else:
        print(f"  - Total files found: {total_files_found}")
        print(f"  - Records skipped: {skipped_count}")
        print(f"  - Records processed successfully: {processed_count}")
        print(f"  - Records with errors: {error_count}")
        if processed_count > 0:
             print(f"  - Cleaned data saved within: {output_dir}")
             print(f"  - Features saved to: {features_csv_path}")
        elif skipped_count == total_files_found and total_files_found > 0:
             print(f"[INFO] All {skipped_count} found records were skipped due to --skip_records setting.")
        elif total_files_found > 0:
             print("[INFO] No records processed successfully (check start_from or errors).")


    # --- Final Comparison (if requested and processing occurred or was skipped) ---
    if args.machine_csv:
        # Check if features file exists *now* (might have been created during run)
        if os.path.exists(features_csv_path):
            print("\n[INFO] Proceeding to comparison with machine measurements...")
            machine_csv_path = os.path.abspath(args.machine_csv)
            run_comparison_only(features_csv_path, machine_csv_path, comparison_output_csv)
        else:
             print(f"\n[WARNING] Machine comparison requested, but features file not found or not created: {features_csv_path}")
             print("[INFO] Skipping comparison.")
    else:
         print("\n[INFO] No machine measurements CSV provided, skipping comparison.")

    print("\n--- Script Finished ---")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main() # Remove the try/except KeyboardInterrupt from here, handled in main()