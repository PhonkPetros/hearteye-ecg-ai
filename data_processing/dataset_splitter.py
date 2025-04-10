import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar library

# Load original dataset (HDF5 file)
original_hdf5_file = "D:/Maike/ecg_data_with_label_and_metadata.h5"

# Open the original HDF5 file and read the data directly into memory
print("Loading dataset...")
with h5py.File(original_hdf5_file, 'r') as f:
    # Load the ECG data and labels into memory (as numpy arrays)
    ecg_data = f['ecg_data'][:]
    labels_data = f['labels'][:]

print(f"Dataset loaded: {ecg_data.shape[0]} samples, {ecg_data.shape[1]} leads, {ecg_data.shape[2]} time points.")

# Perform stratified split: First split into train and remaining (which will be split into validation and test)
print("Performing stratified split (train, validation, test)...")
train_ecg, temp_ecg, train_labels, temp_labels = train_test_split(
    ecg_data, labels_data, test_size=0.2, random_state=42, stratify=labels_data
)

# Split the remaining data into validation and test sets
val_ecg, test_ecg, val_labels, test_labels = train_test_split(
    temp_ecg, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Splits complete: {len(train_ecg)} train, {len(val_ecg)} validation, {len(test_ecg)} test samples.")

# Save these splits into separate HDF5 files
def save_to_hdf5(filename, ecg_data, labels_data):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('ecg_data', data=ecg_data)
        f.create_dataset('labels', data=labels_data)
        print(f"Data saved to {filename}")

# List of files to save
files_to_save = [
    ('D:/Maike/train_data.h5', train_ecg, train_labels),
    ('D:/Maike/val_data.h5', val_ecg, val_labels),
    ('D:/Maike/test_data.h5', test_ecg, test_labels)
]

# Save splits with progress bar
for file_name, ecg, labels in tqdm(files_to_save, desc="Saving splits", unit="file"):
    save_to_hdf5(file_name, ecg, labels)
