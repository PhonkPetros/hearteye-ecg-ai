import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import signal
import wfdb
import traceback
from pathlib import Path

# Import your existing feature extractor
from feature_extraction.extractor import ECGFeatureExtractor

# Global flag to indicate if the process should stop
should_stop = False

def signal_handler(sig, frame):
    """Handle Ctrl+C by setting the stop flag"""
    global should_stop
    print("\nReceived stop signal. Will stop after current record.")
    should_stop = True

def find_ecg_records(cleaned_ecg_dir):
    """
    Find all ECG records in the cleaned ECG directory
    
    Parameters
    ----------
    cleaned_ecg_dir : str
        Path to the cleaned ECG directory
    
    Returns
    -------
    list
        List of record information (subject_id, study_id, record_path)
    """
    records = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(cleaned_ecg_dir):
        for file in files:
            if file.endswith('-cleaned.hea'):
                # Get record path without extension
                record_path = os.path.join(root, file.replace('.hea', ''))
                
                # Extract subject_id from the path (folder starting with 'p')
                subject_id = None
                path_parts = record_path.split(os.sep)
                for part in path_parts:
                    if part.startswith('p') and part[1:].isdigit():
                        subject_id = part[1:]  # Remove 'p' prefix
                        break
                
                # Extract study_id from the path (folder starting with 's')
                study_id = None
                for part in path_parts:
                    if part.startswith('s'):
                        study_id = part
                        break
                
                # If subject_id was found, add to records
                if subject_id:
                    records.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'record_path': record_path,
                        'record_id': os.path.basename(record_path).replace('-cleaned', '')
                    })
    
    return records

def get_already_processed_records(output_csv):
    """
    Get list of already processed record paths from the output CSV
    
    Parameters
    ----------
    output_csv : str
        Path to the output CSV file
    
    Returns
    -------
    set
        Set of already processed record paths
    """
    if not os.path.exists(output_csv):
        return set()
    
    try:
        df = pd.read_csv(output_csv)
        if 'record_path' in df.columns:
            return set(df['record_path'].tolist())
        else:
            return set()
    except Exception as e:
        print(f"Error reading existing results: {str(e)}")
        return set()

def process_records(cleaned_ecg_dir, output_csv, limit=None, compare_after=100):
    """
    Process ECG records one by one, saving after each record
    
    Parameters
    ----------
    cleaned_ecg_dir : str
        Directory containing cleaned ECG files
    output_csv : str
        Path to save extracted features
    limit : int, optional
        Maximum number of records to process
    compare_after : int, optional
        Generate comparison after processing this many records
    
    Returns
    -------
    int
        Number of records processed
    """
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Incremental ECG feature extraction")
    print(f"Press Ctrl+C to stop processing after current record")
    
    # Initialize the feature extractor
    extractor = ECGFeatureExtractor(sampling_rate=500.0)
    
    # Find all ECG records
    all_records = find_ecg_records(cleaned_ecg_dir)
    print(f"Found {len(all_records)} ECG records total")
    
    # Get already processed records
    processed_paths = get_already_processed_records(output_csv)
    print(f"Already processed {len(processed_paths)} records")
    
    # Filter out already processed records
    records_to_process = [r for r in all_records if r['record_path'] not in processed_paths]
    print(f"Will process {len(records_to_process)} new records")
    
    # Limit records if specified
    if limit is not None:
        records_to_process = records_to_process[:min(limit, len(records_to_process))]
        print(f"Limited to {len(records_to_process)} records")
    
    # Track processing statistics
    processed = 0
    errors = 0
    compare_counter = 0
    
    # Process each record
    for i, record in enumerate(records_to_process):
        if should_stop:
            print("Stopping as requested...")
            break
        
        try:
            # Extract record info
            subject_id = record['subject_id']
            study_id = record['study_id']
            record_path = record['record_path']
            record_id = record['record_id']
            
            # Progress information
            print(f"Processing record {i+1}/{len(records_to_process)}: {record_path}")
            
            # Use the extract_from_wfdb method from your extractor
            features = extractor.extract_from_wfdb(record_path)
            
            # Add metadata
            features['subject_id'] = subject_id
            features['study_id'] = study_id
            features['record_id'] = record_id
            features['record_path'] = record_path
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Check if output file exists and has data
            file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
            
            # Save to CSV, append if file exists
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(output_csv, mode=mode, header=header, index=False)
            
            # Update counters
            processed += 1
            compare_counter += 1
            
            # Update progress periodically
            if processed % 10 == 0:
                print(f"Processed {processed} records so far")
            
            # Periodically run comparison if machine_csv is available
            if compare_counter >= compare_after:
                compare_counter = 0
                
                # Check if machine_csv is in the environment
                machine_csv = os.environ.get('MACHINE_CSV')
                comparison_dir = os.environ.get('COMPARISON_DIR')
                
                if machine_csv and comparison_dir and os.path.exists(machine_csv):
                    print(f"Running interim comparison with machine measurements...")
                    try:
                        from comparison_script import compare_with_machine_measurements
                        compare_with_machine_measurements(
                            output_csv,
                            machine_csv,
                            comparison_dir
                        )
                        print(f"Interim comparison complete.")
                    except Exception as compare_e:
                        print(f"Interim comparison failed: {str(compare_e)}")
        
        except Exception as e:
            print(f"Error processing {record_path}: {str(e)}")
            traceback.print_exc()
            errors += 1
    
    print(f"\nProcessing Summary:")
    print(f"Records processed: {processed}")
    print(f"Errors: {errors}")
    
    return processed

def main():
    parser = argparse.ArgumentParser(description='Incrementally process ECG files')
    parser.add_argument('--cleaned_ecg_dir', type=str, required=True,
                        help='Directory containing cleaned ECG files')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save extracted features')
    parser.add_argument('--machine_csv', type=str, default=None,
                        help='Path to CSV file with machine measurements (for comparison)')
    parser.add_argument('--comparison_dir', type=str, default=None,
                        help='Directory to save comparison results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of records to process')
    
    args = parser.parse_args()
    
    # Store machine_csv and comparison_dir in environment for comparison process
    if args.machine_csv:
        os.environ['MACHINE_CSV'] = args.machine_csv
    if args.comparison_dir:
        os.makedirs(args.comparison_dir, exist_ok=True)
        os.environ['COMPARISON_DIR'] = args.comparison_dir
    
    # Start timer
    start_time = time.time()
    print(f"Started incremental ECG processing at {time.ctime(start_time)}")
    
    # Process records
    processed = process_records(
        args.cleaned_ecg_dir,
        args.output_csv,
        args.limit
    )
    
    # End timer
    end_time = time.time()
    duration = (end_time - start_time) / 60.0
    
    print("\nECG Processing Summary:")
    print(f"Total duration: {duration:.2f} minutes")
    print(f"Records processed: {processed}")
    print(f"Results saved to: {args.output_csv}")
    
    # Run final comparison if requested
    if args.machine_csv and args.comparison_dir:
        try:
            from comparison_script import compare_with_machine_measurements
            print("\nRunning final comparison with machine measurements...")
            compare_with_machine_measurements(
                args.output_csv,
                args.machine_csv,
                args.comparison_dir
            )
            print(f"Final comparison saved to: {args.comparison_dir}")
        except Exception as e:
            print(f"Final comparison failed: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())