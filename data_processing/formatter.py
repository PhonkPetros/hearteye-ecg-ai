import wfdb
import numpy as np
import os

base_path = "data/external/cleaned_ecg"
ecg_data = []
total_files_processed = 0

print("Starting ECG data processing...\n")

for patients in os.listdir(base_path):
    patients_path = os.path.join(base_path, patients)

    if not os.path.isdir(patients_path):  # Skip if not a directory
        continue

    for patient in os.listdir(patients_path):
        patient_path = os.path.join(patients_path, patient)

        if not os.path.isdir(patient_path):  # Skip if not a directory
            continue

        print(f"Processing patient: {patient}")

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)

            if not os.path.isdir(study_path):  # Skip if not a directory
                continue

            for file in os.listdir(study_path):
                if file.endswith(".hea"):
                    record_name = os.path.splitext(file)[0]
                    full_record_path = os.path.join(study_path, record_name)

                    try:
                        #read ECG signal
                        record = wfdb.rdrecord(full_record_path)
                        signals = record.p_signal

                        #Check that all signals have same shape
                        if signals.shape[0] == 5000:
                            ecg_data.append(signals)
                            total_files_processed += 1
                            print(f"✔ Processed: {full_record_path} ({total_files_processed})")
                    
                    except Exception as e:
                        print(f"Error reading {full_record_path}: {e}")

print("\nProcessing complete!")
print(f"Total valid ECG records: {total_files_processed}")

# Convert list to NumPy array
ecg_data = np.array(ecg_data)

save_path = "data/external/cleaned-dataset/ecg_data.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save dataset
np.save(save_path, ecg_data)
print(f"ECG data saved to {save_path}. Shape: {ecg_data.shape}")
