import os
import pyedflib
import numpy as np
import wfdb

# === Set paths ===
edf_folder = "data/external/hearteye_edf"
wfdb_output_folder = "data/external/hearteye_edf/wfdb_records_multi_lead"
os.makedirs(wfdb_output_folder, exist_ok=True)

# === Loop through EDF files ===
for filename in os.listdir(edf_folder):
    if filename.lower().endswith(".edf"):
        edf_path = os.path.join(edf_folder, filename)
        record_name = os.path.splitext(filename)[0]
        wfdb_path = os.path.join(wfdb_output_folder, record_name)

        print(f"\n🔁 Converting: {filename}")

        reader = pyedflib.EdfReader(edf_path)
        n_signals = reader.signals_in_file
        fs = reader.getSampleFrequency(0)
        signal_labels = reader.getSignalLabels()

        # === Get lengths of each signal (fixed here)
        lengths = reader.getNSamples()
        print(f"📏 Signal lengths: {lengths}")

        min_len = min(lengths)

        # === Read and align all signals to same length
        signals = np.zeros((min_len, n_signals), dtype=np.float32)
        for i in range(n_signals):
            raw = reader.readSignal(i)
            signals[:, i] = raw[:min_len]

        reader._close()

        # === Save to WFDB ===
        wfdb.wrsamp(
            record_name=record_name,
            fs=int(fs),
            units=['mV'] * n_signals,
            sig_name=signal_labels,
            p_signal=signals,
            write_dir=wfdb_output_folder
        )

        print(f"✅ Saved: {wfdb_path}.dat / .hea")

print("\n🎉 All EDF files successfully converted to WFDB.")
