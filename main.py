import os
import numpy as np
import pandas as pd
import wfdb
import argparse
from pathlib import Path
from data_processing.cleaner import ECGCleaner
from data_processing.loader import ECGLoader
from feature_extraction.extractor import ECGFeatureExtractor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load, clean and extract features from ECG data')
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
    
    # Define features CSV path
    features_csv_path = os.path.join(output_dir, args.features_csv)
    
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
            output_record_name = f"{record_id}-cleaned"  # Use hyphen instead of underscore
            output_record_path = os.path.join(target_dir, output_record_name)
            
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
            
            print(f"[INFO] Saved cleaned data to: {output_record_path} in WFDB format")
            
            # Extract features from cleaned ECG
            print(f"[INFO] Extracting features from cleaned ECG")
            features = extractor.extract_features(cleaned_signals[:, 1] if cleaned_signals.shape[1] >= 2 else cleaned_signals[:, 0])
            
            # Add metadata to features
            features['record_id'] = record_id
            features['patient_id'] = os.path.basename(os.path.dirname(os.path.dirname(hea_path)))
            features['study_id'] = os.path.basename(os.path.dirname(hea_path))
            features['original_path'] = record_base
            features['cleaned_path'] = output_record_path
            
            # Save features to CSV (append mode)
            features_df = pd.DataFrame([features])
            
            # If file doesn't exist, write with header, otherwise append without header
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
    
    print("\n[SUMMARY] Finished ECG processing:")
    print(f"  - Processed {processed} records")
    print(f"  - Skipped {skipped} records")
    print(f"  - Encountered {errors} errors")
    print(f"  - Cleaned data saved to {output_dir} (preserving original folder structure)")
    print(f"  - ECG features saved to {features_csv_path}")

if __name__ == "__main__":
    main()