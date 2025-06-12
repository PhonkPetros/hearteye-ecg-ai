import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Using this for convenience, can also do manually
import os

# --- Configuration ---
# Path to your main CSV file with all labels and fiducial points
FULL_LABELS_CSV_PATH = r"E:\AiModel\normal_ecg_labeled.csv" 

# Output directory for the split CSV files
OUTPUT_SPLIT_DIR = r"E:\AiModel\data_splits" # Make sure this directory exists or can be created
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

# Define the proportions for train, validation, and test sets
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure ratios sum to 1 (approximately, due to potential floating point issues)
if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0):
    raise ValueError("Train, validation, and test ratios must sum to 1.0")

# Random seed for reproducibility of the split
RANDOM_SEED = 42

# --- Main Script ---
def perform_patient_aware_split():
    print(f"Loading full labels CSV from: {FULL_LABELS_CSV_PATH}")
    try:
        full_df = pd.read_csv(FULL_LABELS_CSV_PATH)
        # Basic cleaning similar to the pipeline script
        fiducial_cols_to_check = ['p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end']
        for col in fiducial_cols_to_check:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
        
        # Ensure essential ID columns exist and are not NaN before attempting astype
        id_cols = ['subject_id', 'study_id', 'cart_id']
        for id_col in id_cols:
            if id_col not in full_df.columns:
                raise ValueError(f"Essential ID column '{id_col}' not found in the CSV.")
        
        full_df.dropna(subset=id_cols + fiducial_cols_to_check, inplace=True) 
        
        for id_col in id_cols: # Convert IDs after dropna
            full_df[id_col] = full_df[id_col].astype(np.int64)

        full_df.reset_index(drop=True, inplace=True)
        print(f"Loaded and initially filtered {len(full_df)} records.")
        if full_df.empty:
            print("The loaded CSV is empty after initial filtering. Cannot proceed.")
            return
    except FileNotFoundError:
        print(f"ERROR: Full labels CSV file not found at {FULL_LABELS_CSV_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading or parsing the full labels CSV: {e}")
        return

    # Get unique subject IDs
    unique_subject_ids = full_df['subject_id'].unique()
    print(f"Found {len(unique_subject_ids)} unique subject IDs.")

    if len(unique_subject_ids) < 3: # Need at least one subject for each split
        print("ERROR: Not enough unique subjects to perform a train/validation/test split.")
        return

    # Shuffle the unique subject IDs
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_subject_ids)

    # Split subject IDs
    # First, split into training and (validation + test)
    train_subject_ids, remaining_subject_ids = train_test_split(
        unique_subject_ids,
        test_size=(VALIDATION_RATIO + TEST_RATIO), # Size of the set to be further split
        random_state=RANDOM_SEED
    )

    # Now split the remaining into validation and test
    # Adjust test_size for the second split: TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    if (VALIDATION_RATIO + TEST_RATIO) == 0: # Avoid division by zero if only train split
        val_subject_ids = np.array([])
        test_subject_ids = np.array([])
    else:
        relative_test_size = TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
        if len(remaining_subject_ids) < 2 and relative_test_size > 0 and relative_test_size < 1:
            # Handle cases where remaining_subject_ids is too small for a meaningful split
            # Assign all to validation or test, or handle as per your needs.
            # For simplicity, if only one remaining, it goes to validation. If relative_test_size makes one set empty.
            if np.isclose(relative_test_size, 1.0): # All to test
                 val_subject_ids = np.array([])
                 test_subject_ids = remaining_subject_ids
            elif np.isclose(relative_test_size, 0.0): # All to validation
                 val_subject_ids = remaining_subject_ids
                 test_subject_ids = np.array([])
            else: # Default to putting the single remaining into validation
                 val_subject_ids = remaining_subject_ids
                 test_subject_ids = np.array([])
                 print(f"Warning: Very few subjects for val/test split. {len(remaining_subject_ids)} subject(s) assigned to validation.")

        elif len(remaining_subject_ids) == 0:
            val_subject_ids = np.array([])
            test_subject_ids = np.array([])
        else:
            val_subject_ids, test_subject_ids = train_test_split(
                remaining_subject_ids,
                test_size=relative_test_size,
                random_state=RANDOM_SEED 
            )
    
    print(f"\nSplitting subjects:")
    print(f"  Training set: {len(train_subject_ids)} subjects")
    print(f"  Validation set: {len(val_subject_ids)} subjects")
    print(f"  Test set: {len(test_subject_ids)} subjects")

    # Create DataFrames for each set
    train_df = full_df[full_df['subject_id'].isin(train_subject_ids)].copy()
    val_df = full_df[full_df['subject_id'].isin(val_subject_ids)].copy()
    test_df = full_df[full_df['subject_id'].isin(test_subject_ids)].copy()

    print(f"\nNumber of records in each set:")
    print(f"  Training set: {len(train_df)} records")
    print(f"  Validation set: {len(val_df)} records")
    print(f"  Test set: {len(test_df)} records")
    print(f"  Total records in splits: {len(train_df) + len(val_df) + len(test_df)} (should match original if all subjects assigned)")


    # Save the split DataFrames to new CSV files
    train_csv_path = os.path.join(OUTPUT_SPLIT_DIR, "train_records.csv")
    val_csv_path = os.path.join(OUTPUT_SPLIT_DIR, "val_records.csv")
    test_csv_path = os.path.join(OUTPUT_SPLIT_DIR, "test_records.csv")

    try:
        train_df.to_csv(train_csv_path, index=False)
        print(f"\nTraining records saved to: {train_csv_path}")
        val_df.to_csv(val_csv_path, index=False)
        print(f"Validation records saved to: {val_csv_path}")
        test_df.to_csv(test_csv_path, index=False)
        print(f"Test records saved to: {test_csv_path}")
    except Exception as e_save:
        print(f"ERROR saving split CSV files: {e_save}")

if __name__ == "__main__":
    perform_patient_aware_split()

