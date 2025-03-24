import wfdb
import numpy as np
import os

base_path = "C:\Users\maike\hearteye-ecg-ai\data\external\cleaned_ecg"
ecg_data = []

for patients in os.listdir(base_path):
    patients_path = os.path.join(base_path, patients)

    for patient in os.listdir(patients_path):
        patient_path = os.path.join(patients_path, patient)

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)

            for file in os.listdir(study_path):
                if file.endswith(".hea"):
                    record_name = os.path.splitext(file)[0]
                    full record_path = os.join(study_path, record_name)

                    #read ECG signal
                    record = wfdb.rdrecord(full_recors_path)
                    signals = record.p_signal

                    #Check that all signals have same shape
                    if signals.shape[0] == 5000:
                        ecg_data.append(signals)

ecg_data = np.array(ecg_data)
np.save("C:\Users\maike\hearteye-ecg-ai\data\external\cleaned-dataset/ecg_data.npy", ecg_data)
