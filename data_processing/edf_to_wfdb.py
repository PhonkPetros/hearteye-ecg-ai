import pyedflib
import numpy as np
from scipy.signal import resample
import wfdb
import os

edf_dir = "C:/Users/maike/hearteye-ecg-ai/data/external/hearteye_edf"
output_dir = "C:/Users/maike/hearteye-ecg-ai/data/external/hearteye_wfdb"
target_samples = 5000
target_fs = 500

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop over all .edf files
for filename in os.listdir(edf_dir):
    if filename.lower().endswith(".edf"):
        edf_path = os.path.join(edf_dir, filename)
        record_name = os.path.splitext(filename)[0] + "_fixedlen"
        
        # Create a subfolder for each ECG file
        record_output_dir = os.path.join(output_dir, record_name)
        os.makedirs(record_output_dir, exist_ok=True)

        # Remove any existing desktop.ini file in the output directory
        desktop_ini_path = os.path.join(record_output_dir, 'desktop.ini')
        if os.path.exists(desktop_ini_path):
            os.remove(desktop_ini_path)


        print(f"Processing: {filename}")

        try:
            f = pyedflib.EdfReader(edf_path)
            n_signals = f.signals_in_file
            signal_labels = f.getSignalLabels()

            signals = []
            for i in range(n_signals):
                sig = f.readSignal(i)
                sig_resampled = resample(sig, target_samples)
                signals.append(sig_resampled)

            f.close()

            # Convert to 2D numpy array (shape: [n_samples, n_channels])
            signals_array = np.array(signals).T

            # Save the record in its own subfolder
            wfdb.wrsamp(
                record_name=record_name,
                write_dir=record_output_dir,
                fs=target_fs,
                units=['mV'] * n_signals,
                sig_name=signal_labels,
                p_signal=signals_array
            )

            print(f"Saved: {record_name} ({signals_array.shape[0]} samples x {signals_array.shape[1]} signals)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
