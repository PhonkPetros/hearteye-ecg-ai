import os
import wfdb
import pandas as pd

class ECGLoader:
    """
    Loads .hea/.dat files using WFDB, plus an optional CSV of labels.
    """

    def __init__(self, base_dir, csv_path):
        """
        Parameters
        ----------
        base_dir : str
            Directory containing the .hea/.dat files (possibly in subfolders).
        csv_path : str
            Path to a CSV file containing ICD or diagnostic labels (optional usage).
        """
        self.base_dir = base_dir
        self.csv_path = csv_path
        self.df_labels = None  # Will be loaded if needed

    def load_csv_labels(self):
        """
        Load label info from the CSV. Adjust column names or logic as needed.
        """
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] CSV path does not exist: {self.csv_path}")
            return None
        print(f"[DEBUG] Loading labels from CSV: {self.csv_path}")
        self.df_labels = pd.read_csv(self.csv_path)
        print(f"[DEBUG] Loaded {len(self.df_labels)} rows from CSV.")
        return self.df_labels

    def iter_ecg_records(self):
        """
        Generator that walks through self.base_dir, yielding (.hea_path, .dat_path).
        Skips zero-sized or missing .dat files.
        """
        print(f"[DEBUG] Scanning base directory for .hea/.dat pairs: {self.base_dir}")
        for root, dirs, files in os.walk(self.base_dir):
            for f in files:
                if f.lower().endswith(".hea"):
                    hea_path = os.path.join(root, f)
                    dat_path = os.path.splitext(hea_path)[0] + ".dat"
                    if not os.path.exists(dat_path):
                        # matching .dat not found
                        continue
                    try:
                        if os.path.getsize(hea_path) == 0 or os.path.getsize(dat_path) == 0:
                            continue
                    except OSError:
                        continue
                    yield (hea_path, dat_path)

    def load_ecg_record(self, record_base):
        """
        Loads ECG data from WFDB given the record base name (full path without extension).
        Returns:
          signals (np.ndarray) shape=(N,Ch),
          fields (dict) from WFDB (contains 'fs', etc.).
        """
        try:
            print(f"[DEBUG] wfdb.rdsamp => Loading record: {record_base}")
            signals, fields = wfdb.rdsamp(record_base)
            print(f"   [DEBUG] Loaded signals shape: {signals.shape}, Fields keys: {list(fields.keys())}")
            return signals, fields
        except Exception as e:
            print(f"[ERROR] Could not load record {record_base}: {e}")
            raise RuntimeError(
                f"Failed to load record: {record_base}. Check if .dat/.hea are accessible."
            ) from e