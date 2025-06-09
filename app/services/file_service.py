import wfdb
import numpy as np
import pyedflib
import os
import re
import logging
from .ecg_processing_service import ECGProcessingService

class FileHandlingService:
    @staticmethod
    def load_and_clean_all_leads(wfdb_path):
        """
        Load and clean all leads from a WFDB record
        """
        try:
            if not os.path.isdir(wfdb_path):
                wfdb_basename = os.path.splitext(wfdb_path)[0]
            else:
                hea_file = next(
                    (f for f in os.listdir(wfdb_path) if f.lower().endswith(".hea")), None
                )
                if not hea_file:
                    raise FileNotFoundError(f"No .hea file found in directory {wfdb_path}")
                wfdb_basename = os.path.join(wfdb_path, os.path.splitext(hea_file)[0])

            signals, fs, record = ECGProcessingService.read_wfdb_record(wfdb_basename)
            lead_names = (
                record.sig_name
                if record.sig_name
                else [f"Lead_{i}" for i in range(signals.shape[1])]
            )

            return {"cleaned_signals": signals, "fs": fs, "lead_names": lead_names}
        except Exception as e:
            logging.error(
                f"Error in load_and_clean_all_leads for path {wfdb_path}: {e}",
                exc_info=True,
            )
            raise

    @staticmethod
    def convert_edf_to_wfdb(edf_path, output_dir=None):
        """
        Converts an edf file to wfdb
        """

        def sanitize_filename(name: str) -> str:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
            sanitized = sanitized.strip("_")
            return sanitized or "default_record_name"

        try:
            if output_dir is None:
                output_dir = os.path.dirname(edf_path)

            filename_base = os.path.splitext(os.path.basename(edf_path))[0]
            record_name = sanitize_filename(filename_base)

            edf_reader = pyedflib.EdfReader(edf_path)
            num_signals = edf_reader.signals_in_file
            fs = edf_reader.getSampleFrequency(0)
            signal_labels = edf_reader.getSignalLabels()
            signals = [edf_reader.readSignal(i) for i in range(num_signals)]
            edf_reader.close()
            del edf_reader

            # Trim signals to shortest length
            min_len = min(len(s) for s in signals)
            signals_trimmed = [s[:min_len] for s in signals]

            # Save as WFDB using record_name and write_dir separately
            wfdb.wrsamp(
                record_name,
                fs=fs,
                units=["uV"] * num_signals,
                sig_name=signal_labels,
                p_signal=np.column_stack(signals_trimmed),
                write_dir=output_dir,
            )

            return os.path.join(output_dir, record_name)

        except Exception as e:
            raise Exception(f"EDF to WFDB conversion failed: {e}")
