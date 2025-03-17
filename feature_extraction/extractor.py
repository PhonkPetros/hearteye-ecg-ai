import numpy as np
import neurokit2 as nk

class ECGExtractor:
    """
    Uses NeuroKit2 to delineate ECG and extract durations: QRS, QT, PQ, P-wave.
    """

    def __init__(self, sampling_rate=None, clean_method='neurokit', delineate_method='dwt'):
        """
        :param sampling_rate: If None, will rely on the data's 'fs'.
        :param clean_method: Method for nk.ecg_clean (e.g. 'neurokit', 'biosppy', etc.).
        :param delineate_method: Method for nk.ecg_delineate (e.g. 'dwt', 'cwt', 'peak', 'prominence').
        """
        self.sampling_rate = sampling_rate
        self.clean_method = clean_method
        self.delineate_method = delineate_method

    def preprocess(self, signal, fs):
        """
        NeuroKit2-based cleaning (bandpass, etc.). If it fails, returns original.
        """
        print(f"[DEBUG] ECGExtractor.preprocess: cleaning with NeuroKit2 method={self.clean_method}")
        sampling_rate = fs or self.sampling_rate
        if sampling_rate is None:
            raise ValueError("Sampling rate required if not found in WFDB record fields.")
        before_min, before_max = signal.min(), signal.max()
        print(f"   [DEBUG] signal min={before_min:.3f}, max={before_max:.3f}, fs={sampling_rate}")

        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method=self.clean_method)
        except Exception as e:
            print(f"   [ERROR] nk.ecg_clean failed: {e}, using original signal.")
            cleaned = signal
        after_min, after_max = cleaned.min(), cleaned.max()
        print(f"   [DEBUG] cleaned signal min={after_min:.3f}, max={after_max:.3f}")
        return cleaned, sampling_rate

    def extract_durations(self, ecg_signal, fs):
        """
        Main feature extraction:
         - QRS duration
         - QT interval
         - PQ interval
         - P-wave duration
        Returns a dict with these keys. Values are averaged across all detected beats.
        """
        durations = {
            "QRS_duration": None,
            "QT_duration": None,
            "PQ_interval": None,
            "P_duration": None,
        }
        print("[DEBUG] ECGExtractor.extract_durations: Starting feature extraction.")
        ecg_cleaned, sampling_rate = self.preprocess(ecg_signal, fs)

        # R-peak detection
        try:
            print("   [DEBUG] Running nk.ecg_peaks...")
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            rpeaks_indices = rpeaks.get("ECG_R_Peaks", [])
            print(f"   [DEBUG] Found {len(rpeaks_indices)} R-peaks.")
        except Exception as e:
            print(f"   [ERROR] nk.ecg_peaks failed: {e}")
            rpeaks_indices = []

        if len(rpeaks_indices) == 0:
            print("   [DEBUG] No R-peaks found, skipping wave delineation.")
            return durations  # all None => can't compute intervals

        # Wave delineation
        try:
            print("   [DEBUG] Running nk.ecg_delineate...")
            _, waves = nk.ecg_delineate(
                ecg_cleaned,
                rpeaks=rpeaks,
                sampling_rate=sampling_rate,
                method=self.delineate_method,
                check=True
            )
            print("   [DEBUG] Delineation keys:", list(waves.keys()))
        except Exception as e:
            print(f"   [ERROR] nk.ecg_delineate failed: {e}")
            return durations

        qrs_durations = []
        qt_intervals = []
        pq_intervals = []
        p_durations = []

        for idx, r_idx in enumerate(rpeaks_indices):
            # Onset/offset for QRS
            R_onset_idx = None
            R_offset_idx = None

            if "ECG_R_Onsets" in waves:
                # R_onsets <= r_idx
                R_onsets_list = waves["ECG_R_Onsets"]
                R_onset_idx = max([on for on in R_onsets_list if on <= r_idx], default=None)
            if "ECG_R_Offsets" in waves:
                R_offsets_list = waves["ECG_R_Offsets"]
                R_offset_idx = min([off for off in R_offsets_list if off >= r_idx], default=None)

            # Fallback to Q or S if needed
            if R_onset_idx is None and "ECG_Q_Peaks" in waves:
                Q_list = waves["ECG_Q_Peaks"]
                R_onset_idx = max([q for q in Q_list if q < r_idx], default=None)
            if R_offset_idx is None and "ECG_S_Peaks" in waves:
                S_list = waves["ECG_S_Peaks"]
                R_offset_idx = min([s for s in S_list if s > r_idx], default=None)

            # Onset/offset for P
            P_onset_idx = None
            P_offset_idx = None
            if "ECG_P_Onsets" in waves:
                P_onset_idx = max([p for p in waves["ECG_P_Onsets"] if p < r_idx], default=None)
            if "ECG_P_Offsets" in waves:
                P_offset_idx = max([p for p in waves["ECG_P_Offsets"] if p < r_idx], default=None)

            # T wave offset
            T_offset_idx = None
            if "ECG_T_Offsets" in waves:
                T_offset_idx = min([t for t in waves["ECG_T_Offsets"] if t > r_idx], default=None)

            # Compute durations (ms)
            # QRS
            if R_onset_idx is not None and R_offset_idx is not None:
                qrs_ms = (R_offset_idx - R_onset_idx) * (1000.0 / sampling_rate)
                qrs_durations.append(qrs_ms)
            # QT
            if R_onset_idx is not None and T_offset_idx is not None:
                qt_ms = (T_offset_idx - R_onset_idx) * (1000.0 / sampling_rate)
                qt_intervals.append(qt_ms)
            # PQ
            if P_onset_idx is not None and R_onset_idx is not None:
                pq_ms = (R_onset_idx - P_onset_idx) * (1000.0 / sampling_rate)
                pq_intervals.append(pq_ms)
            # P duration
            if P_onset_idx is not None and P_offset_idx is not None:
                p_ms = (P_offset_idx - P_onset_idx) * (1000.0 / sampling_rate)
                p_durations.append(p_ms)

        # Average across beats
        if qrs_durations:
            durations["QRS_duration"] = float(np.nanmean(qrs_durations))
        if qt_intervals:
            durations["QT_duration"] = float(np.nanmean(qt_intervals))
        if pq_intervals:
            durations["PQ_interval"] = float(np.nanmean(pq_intervals))
        if p_durations:
            durations["P_duration"] = float(np.nanmean(p_durations))

        # Print results for debugging
        print("[DEBUG] Extraction results so far:")
        for k, v in durations.items():
            print(f"   {k}: {v}")
        return durations