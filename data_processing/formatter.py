import wfdb
import numpy as np
import os

base_path = "data/external/cleaned_ecg"
save_dir = "data/external/cleaned-dataset"
os.makedirs(save_dir, exist_ok=True)

total_files_processed = 0

for patients in os.listdir(base_path):
    for patient in os.listdir(os.path.join(base_path, patients)):
        for study in os.listdir(os.path.join(base_path, patients, patient)):
            study_path = os.path.join(base_path, patients, patient, study)

            for file in os.listdir(study_path):
                if file.endswith(".hea"):
                    record_name = os.path.splitext(file)[0]
                    full_record_path = os.path.join(study_path, record_name)

                    try:
                        record = wfdb.rdrecord(full_record_path)
                        signals = record.p_signal  # Defaults to float64

                        if signals.shape[0] == 5000:
                            study_id = study  # Use study ID as filename
                            save_path = os.path.join(save_dir, f"{study_id}.npy")
                            
                            np.save(save_path, signals)  # Keeps float64 precision
                            total_files_processed += 1
                            print(f"✔ Saved: {save_path} ({total_files_processed})")

                    except Exception as e:
                        print(f"Error reading {full_record_path}: {e}")

print("\nProcessing complete!")
print(f"Total valid ECG records saved: {total_files_processed}")
