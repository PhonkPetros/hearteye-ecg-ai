import numpy as np
import neurokit2 as nk
import pandas as pd

class ECGExtractor:
    """
    Uses NeuroKit2 to delineate ECG and extract durations: QRS, QT, PQ, P-wave.
    Contains improved validity checks and lead selection.
    """

    def __init__(self, sampling_rate=None, clean_method='neurokit', delineate_method='dwt'):
        """
        :param sampling_rate: If None, will rely on the data's 'fs'.
        :param clean_method: Method for nk.ecg_clean (e.g. 'neurokit', 'biosppy', etc.).
        :param delineate_method: Method for nk.ecg_delineate (e.g. 'dwt', 'cwt', 'peak', 'prominence').
        """
        self.sampling_rate = sampling_rate
        self.clean_method = clean_method
        # IMPROVEMENT: Try more accurate wavelet-based methods like 'cwt' instead of 'dwt'
        self.delineate_method = delineate_method
        
        # Physiological limits for interval durations in ms
        # IMPROVEMENT: Make these limits more nuanced (e.g., heart-rate dependent)
        self.limits = {
            'QRS_duration': {'min': 50, 'max': 200},
            'QT_duration': {'min': 300, 'max': 600},
            'PQ_interval': {'min': 80, 'max': 250},
            'P_duration': {'min': 50, 'max': 150}
        }

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
            # IMPROVEMENT: Add more sophisticated noise filtering techniques
            # Consider adaptive filtering or wavelet denoising here
            cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method=self.clean_method)
        except Exception as e:
            print(f"   [ERROR] nk.ecg_clean failed: {e}, using original signal.")
            cleaned = signal
            
        after_min, after_max = cleaned.min(), cleaned.max()
        print(f"   [DEBUG] cleaned signal min={after_min:.3f}, max={after_max:.3f}")
        return cleaned, sampling_rate

    def is_valid_duration(self, duration, feature_name):
        """
        Check if a measured duration is physiologically plausible.
        """
        if duration is None:
            return False
            
        limits = self.limits.get(feature_name)
        if limits is None:
            return True
            
        return limits['min'] <= duration <= limits['max']

    def select_best_lead(self, signals, fs):
        """
        Analyze all leads to find the best one for interval measurement.
        Returns index of best lead and a quality score (0-1).
        """
        if signals.ndim < 2:
            return 0, 1.0  # Only one lead available
            
        num_leads = signals.shape[1]
        lead_scores = []
        
        print(f"[DEBUG] Analyzing {num_leads} leads to find the best one...")
        
        for i in range(num_leads):
            lead_signal = signals[:, i]
            
            # Try to clean the signal
            try:
                cleaned, _ = self.preprocess(lead_signal, fs)
            except Exception:
                lead_scores.append(0)
                continue
                
            # Try to find R-peaks
            try:
                # IMPROVEMENT: Use more accurate R-peak detection algorithms
                # Consider Pan-Tompkins or ensemble methods
                _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
                rpeaks_indices = rpeaks.get("ECG_R_Peaks", [])
                r_peak_count = len(rpeaks_indices)
                
                if r_peak_count == 0:
                    lead_scores.append(0)
                    continue
                
                # Calculate signal quality: amplitude and regularity of R-peaks
                # IMPROVEMENT: Add more sophisticated signal quality metrics
                # Consider SNR, baseline stability, and power spectrum analysis
                if r_peak_count >= 2:
                    # 1. R-peak amplitude relative to signal
                    r_amplitudes = [cleaned[idx] for idx in rpeaks_indices]
                    signal_std = np.std(cleaned)
                    amplitude_score = min(1.0, np.mean(np.abs(r_amplitudes)) / (4 * signal_std))
                    
                    # 2. Regularity of R-R intervals
                    rr_intervals = np.diff(rpeaks_indices)
                    regularity_score = 1.0 - min(1.0, np.std(rr_intervals) / np.mean(rr_intervals))
                    
                    # Combined score
                    lead_scores.append(0.5 * amplitude_score + 0.5 * regularity_score)
                else:
                    lead_scores.append(0.1)  # Only one R-peak, low score
            except Exception:
                lead_scores.append(0)
        
        if not lead_scores or max(lead_scores) == 0:
            return 0, 0  # Default to first lead if all failed
        
        best_lead = np.argmax(lead_scores)
        print(f"[DEBUG] Best lead: {best_lead} with quality score {lead_scores[best_lead]:.2f}")
        return best_lead, lead_scores[best_lead]

    def extract_durations(self, ecg_signals, fs):
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
        
        # If multiple leads provided, find the best one
        if ecg_signals.ndim > 1 and ecg_signals.shape[1] > 1:
            best_lead, quality = self.select_best_lead(ecg_signals, fs)
            if quality < 0.3:
                print(f"[WARNING] All leads have poor quality, results may be unreliable")
            
            ecg_signal = ecg_signals[:, best_lead]
            print(f"[DEBUG] Using lead {best_lead} for analysis")
        else:
            # Single lead provided
            ecg_signal = ecg_signals.flatten() if ecg_signals.ndim > 1 else ecg_signals
        
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
            # IMPROVEMENT: Replace NeuroKit2 delineation with more accurate algorithms
            # Consider using specialized ECG libraries or implementation of more advanced algorithms
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

            # IMPROVEMENT: This wave boundary detection logic could be enhanced
            # Consider statistical or template-based approaches for more consistent detection
            if "ECG_R_Onsets" in waves:
                R_onsets_list = waves["ECG_R_Onsets"]
                R_onset_candidates = [on for on in R_onsets_list if on <= r_idx]
                if R_onset_candidates:
                    R_onset_idx = max(R_onset_candidates)
                    
            if "ECG_R_Offsets" in waves:
                R_offsets_list = waves["ECG_R_Offsets"]
                R_offset_candidates = [off for off in R_offsets_list if off >= r_idx]
                if R_offset_candidates:
                    R_offset_idx = min(R_offset_candidates)

            # Fallback to Q or S if needed
            if R_onset_idx is None and "ECG_Q_Peaks" in waves:
                Q_list = waves["ECG_Q_Peaks"]
                Q_candidates = [q for q in Q_list if q < r_idx]
                if Q_candidates:
                    R_onset_idx = max(Q_candidates)
                    
            if R_offset_idx is None and "ECG_S_Peaks" in waves:
                S_list = waves["ECG_S_Peaks"]
                S_candidates = [s for s in S_list if s > r_idx]
                if S_candidates:
                    R_offset_idx = min(S_candidates)

            # Onset/offset for P
            P_onset_idx = None
            P_offset_idx = None
            
            if "ECG_P_Onsets" in waves:
                P_onset_candidates = [p for p in waves["ECG_P_Onsets"] if p < r_idx]
                if P_onset_candidates:
                    P_onset_idx = max(P_onset_candidates)
                    
            if "ECG_P_Offsets" in waves:
                P_offset_candidates = [p for p in waves["ECG_P_Offsets"] if p < r_idx]
                if P_offset_candidates:
                    P_offset_idx = max(P_offset_candidates)

            # T wave offset
            T_offset_idx = None
            
            if "ECG_T_Offsets" in waves:
                T_offset_candidates = [t for t in waves["ECG_T_Offsets"] if t > r_idx]
                if T_offset_candidates:
                    T_offset_idx = min(T_offset_candidates)

            # Compute durations (ms) with validity checks
            # IMPROVEMENT: Instead of just rejecting outliers, consider weighted averaging 
            # or beat confidence scores based on signal quality
            # QRS
            if R_onset_idx is not None and R_offset_idx is not None:
                qrs_ms = (R_offset_idx - R_onset_idx) * (1000.0 / sampling_rate)
                if self.limits['QRS_duration']['min'] <= qrs_ms <= self.limits['QRS_duration']['max']:
                    qrs_durations.append(qrs_ms)
                else:
                    print(f"   [DEBUG] Rejected implausible QRS duration: {qrs_ms:.2f}ms for beat {idx}")
                    
            # QT
            if R_onset_idx is not None and T_offset_idx is not None:
                qt_ms = (T_offset_idx - R_onset_idx) * (1000.0 / sampling_rate)
                if self.limits['QT_duration']['min'] <= qt_ms <= self.limits['QT_duration']['max']:
                    qt_intervals.append(qt_ms)
                else:
                    print(f"   [DEBUG] Rejected implausible QT interval: {qt_ms:.2f}ms for beat {idx}")
                    
            # PQ
            if P_onset_idx is not None and R_onset_idx is not None:
                pq_ms = (R_onset_idx - P_onset_idx) * (1000.0 / sampling_rate)
                if self.limits['PQ_interval']['min'] <= pq_ms <= self.limits['PQ_interval']['max']:
                    pq_intervals.append(pq_ms)
                else:
                    print(f"   [DEBUG] Rejected implausible PQ interval: {pq_ms:.2f}ms for beat {idx}")
                    
            # P duration
            if P_onset_idx is not None and P_offset_idx is not None:
                p_ms = (P_offset_idx - P_onset_idx) * (1000.0 / sampling_rate)
                if self.limits['P_duration']['min'] <= p_ms <= self.limits['P_duration']['max']:
                    p_durations.append(p_ms)
                else:
                    print(f"   [DEBUG] Rejected implausible P duration: {p_ms:.2f}ms for beat {idx}")

        # Average across beats
        # IMPROVEMENT: Use median instead of mean to reduce impact of outliers
        # or implement more advanced statistical methods
        if qrs_durations:
            durations["QRS_duration"] = float(np.mean(qrs_durations))
        if qt_intervals:
            durations["QT_duration"] = float(np.mean(qt_intervals))
        if pq_intervals:
            durations["PQ_interval"] = float(np.mean(pq_intervals))
        if p_durations:
            durations["P_duration"] = float(np.mean(p_durations))

        # Print results for debugging
        print("[DEBUG] Extraction results:")
        valid_count = sum(1 for v in durations.values() if v is not None)
        print(f"   Valid measurements: {valid_count}/4")
        for k, v in durations.items():
            print(f"   {k}: {v}")
            
        return durations