import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from pathlib import Path
from data_processing.cleaner import ECGCleaner
from data_processing.loader import ECGLoader

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load and clean ECG data')
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
    args = parser.parse_args()
    
    # Ask user in terminal how many records to skip
    skip_input = input("How many records would you like to skip? [0]: ")
    skip_count = int(skip_input) if skip_input.strip() else 0
    print(f"[INFO] Will skip {skip_count} records")
    
    # Create absolute paths
    base_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Cleaned data will be saved to: {output_dir} (preserving original folder structure)")
    
    # Initialize loader and cleaner
    loader = ECGLoader(base_dir, args.labels_csv)
    cleaner = ECGCleaner(fs=args.sampling_rate)
    
    print(f"[DEBUG] ECGLoader initialized with base_dir: {base_dir}")
    print(f"[DEBUG] ECGCleaner initialized with sampling rate: {args.sampling_rate} Hz")
    
    # If labels CSV is provided, load it
    if args.labels_csv:
        labels_df = loader.load_csv_labels()
    
    # Process each ECG record
    found_records = False
    count = 0
    processed = 0
    errors = 0
    
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
            output_record_name = f"{record_id}-cleaned"  # Use hyphen instead of underscore
            
            # Save as WFDB format with correct record name and write_dir parameters
            wfdb.wrsamp(
                record_name=output_record_name,  # Just the record name without full path
                fs=fields['fs'],
                units=fields['units'],
                sig_name=fields['sig_name'],
                p_signal=cleaned_signals,
                fmt=['16'] * cleaned_signals.shape[1],  # 16-bit format
                write_dir=target_dir  # Specify the directory separately
            )
            
            print(f"[INFO] Saved cleaned data to: {os.path.join(target_dir, output_record_name)} in WFDB format")
            processed += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process record {hea_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    if not found_records:
        print("[WARNING] No valid ECG records found! Check the input directory structure.")
        # Check if directory exists
        if not os.path.exists(base_dir):
            print(f"[ERROR] Input directory does not exist: {base_dir}")
        else:
            # List files in directory to debug
            print(f"[DEBUG] Files in input directory:")
            for root, dirs, files in os.walk(base_dir):
                print(f"  Directory: {root}")
                for file in files:
                    print(f"    - {file}")
    elif processed == 0:
        print(f"[WARNING] Skipped all {skipped} records found. Check your skip count and start_from parameters.")
    
    print("\n[SUMMARY] Finished ECG cleaning process:")
    print(f"  - Processed {processed} records")
    print(f"  - Skipped {skipped} records")
    print(f"  - Encountered {errors} errors")
    print(f"  - Cleaned data saved to {output_dir} (preserving original folder structure)")

if __name__ == "__main__":
    main()
    