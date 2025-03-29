import os
import numpy as np

# Path to your ECG data directory (same as in the previous script)
base_path = "data/external/cleaned_ecg"
ids = []  # List to hold the study part IDs

print("Rebuilding study IDs from the directory structure...\n")

# Traverse the directory structure and generate the study IDs
for patients in os.listdir(base_path):
    patients_path = os.path.join(base_path, patients)

    if not os.path.isdir(patients_path):  # Skip if not a directory
        continue

    for patient in os.listdir(patients_path):
        patient_path = os.path.join(patients_path, patient)

        if not os.path.isdir(patient_path):  # Skip if not a directory
            continue

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)

            if not os.path.isdir(study_path):  # Skip if not a directory
                continue

            for file in os.listdir(study_path):
                if file.endswith(".hea"):
                    # Extract just the study part from the directory (e.g., 's46124511')
                    study_id = os.path.basename(study_path)  # Get the study part (e.g., 's46124511')
                    ids.append(study_id)

# Convert the list of IDs into a NumPy array
ids = np.array(ids)

# Save the IDs as a .npy file
ids_save_path = "data/external/cleaned-dataset/ids.npy"
os.makedirs(os.path.dirname(ids_save_path), exist_ok=True)
np.save(ids_save_path, ids)

print(f"Study IDs saved to {ids_save_path}. Shape: {ids.shape}")
