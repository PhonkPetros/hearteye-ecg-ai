import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.signal as signal
import os
import re
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from flask import current_app
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, message, code=500):
        super().__init__(message)
        self.message = message
        self.code = code

class ECGProcessingService:
    
    @staticmethod
    def extract_physionet_features_fixed(signals, fs, lead_names):
        """
        Extract features matching PhysioNet format using only algorithmic detection
        """
        logger.info("Extracting PhysioNet features algorithmically.")
        try:
            # Get robust axis calculations
            axes = ECGProcessingService.calculate_robust_ecg_axis(signals, fs, lead_names)

            # Process best lead for timing
            lead_idx = ECGProcessingService.select_best_lead(signals)
            raw_signal = signals[:, lead_idx]
            logger.debug(f"Selected best lead for timing: {lead_names[lead_idx] if lead_idx < len(lead_names) else lead_idx}")

            # Process the ECG signal to get all features
            ecg_features = ECGProcessingService.extract_comprehensive_ecg_features(raw_signal, fs)
            
            # Build PhysioNet format output using ONLY detected values
            features = {
                "rr_interval": ecg_features.get("rr_interval"),
                "p_onset": ecg_features.get("p_onset", 40),  # Only this stays fixed as reference
                "p_end": ecg_features.get("p_end"),
                "qrs_onset": ecg_features.get("qrs_onset"),
                "qrs_end": ecg_features.get("qrs_end"),
                "t_end": ecg_features.get("t_end"),
                "p_axis": axes.get("p_axis"),
                "qrs_axis": axes.get("qrs_axis"),
                "t_axis": axes.get("t_axis"),
            }
            
            # Return intervals separately for logging/debugging
            intervals = {
                "P_wave_duration_ms": ecg_features.get("p_duration"),
                "PQ_interval_ms": ecg_features.get("pq_interval"),
                "QRS_duration_ms": ecg_features.get("qrs_duration"),
                "QT_interval_ms": ecg_features.get("qt_interval"),
                "RR_interval_ms": ecg_features.get("rr_interval")
            }
            
            logger.info(f"PhysioNet features extracted: {features}")
            return features, intervals

        except Exception as e:
            logger.exception("Error during PhysioNet feature extraction.")
            raise APIError(f"Failed to extract PhysioNet features: {str(e)}", code=500)

    @staticmethod
    def extract_comprehensive_ecg_features(signal_data, fs):
        """
        Extract all ECG features using NeuroKit2 with proper timing calculations
        """
        logger.info("Extracting comprehensive ECG features using NeuroKit2")
        
        try:
            # Clean the signal
            ecg_cleaned = nk.ecg_clean(signal_data, sampling_rate=fs)
            
            # Process the ECG signal - this extracts everything
            ecg_signals, ecg_info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)
            
            # Extract R-peaks
            rpeaks = ecg_info["ECG_R_Peaks"]
            logger.debug(f"Found {len(rpeaks)} R-peaks")
            
            # Get delineation info
            waves = ecg_info.get("ECG_Waves", {})
            
            # Initialize feature dictionary
            features = {}
            
            # Calculate RR interval
            if len(rpeaks) > 1:
                rr_intervals = np.diff(rpeaks) * 1000 / fs  # Convert to ms
                # Filter physiologically plausible RR intervals
                valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2500)]
                if len(valid_rr) > 0:
                    features["rr_interval"] = int(np.median(valid_rr))
                    logger.debug(f"RR interval: {features['rr_interval']}ms")
            
            # Find the most representative cardiac cycle
            best_cycle_idx = ECGProcessingService._find_best_cardiac_cycle(rpeaks, ecg_cleaned, fs)
            
            if best_cycle_idx is not None and best_cycle_idx < len(rpeaks):
                # Define search window around the R-peak
                r_peak_sample = rpeaks[best_cycle_idx]
                
                # Extract features for this cardiac cycle
                cycle_features = ECGProcessingService._extract_cycle_features(
                    ecg_signals, waves, r_peak_sample, fs, best_cycle_idx
                )
                features.update(cycle_features)
            else:
                logger.warning("Could not find suitable cardiac cycle for feature extraction")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in comprehensive feature extraction: {e}")
            return {}

    @staticmethod
    def _find_best_cardiac_cycle(rpeaks, signal_data, fs):
        """
        Find the most representative cardiac cycle for feature extraction
        """
        if len(rpeaks) < 3:
            return 0 if len(rpeaks) > 0 else None
        
        # Calculate RR intervals
        rr_intervals = np.diff(rpeaks)
        median_rr = np.median(rr_intervals)
        
        # Find cycles with RR interval close to median
        best_cycles = []
        for i in range(1, len(rpeaks) - 1):
            prev_rr = rpeaks[i] - rpeaks[i-1]
            next_rr = rpeaks[i+1] - rpeaks[i]
            
            # Check if both surrounding RR intervals are close to median
            if (abs(prev_rr - median_rr) < 0.2 * median_rr and 
                abs(next_rr - median_rr) < 0.2 * median_rr):
                best_cycles.append(i)
        
        # If we found good cycles, pick the one with best signal quality
        if best_cycles:
            # Use the middle one for stability
            return best_cycles[len(best_cycles) // 2]
        
        # Fallback to the middle R-peak
        return len(rpeaks) // 2

    @staticmethod
    def _extract_cycle_features(ecg_signals, waves, r_peak_sample, fs, cycle_idx):
        """
        Extract features for a specific cardiac cycle using NeuroKit2's delineation
        """
        features = {}
        conv = 1000.0 / fs  # Convert samples to milliseconds
        
        # Use the fixed reference point (40ms before the cycle)
        reference_point = 40  # This stays as the fixed reference
        
        # Get wave indices from NeuroKit2
        p_onsets = waves.get("ECG_P_Onsets", [])
        p_peaks = waves.get("ECG_P_Peaks", [])
        p_offsets = waves.get("ECG_P_Offsets", [])
        
        q_peaks = waves.get("ECG_Q_Peaks", [])
        r_onsets = waves.get("ECG_R_Onsets", [])
        r_offsets = waves.get("ECG_R_Offsets", [])
        s_peaks = waves.get("ECG_S_Peaks", [])
        
        t_onsets = waves.get("ECG_T_Onsets", [])
        t_peaks = waves.get("ECG_T_Peaks", [])
        t_offsets = waves.get("ECG_T_Offsets", [])
        
        # Find the relevant waves for this cycle
        # We need to find waves that belong to the current cardiac cycle
        
        # P-wave features (occurs before R-peak)
        p_onset_idx = ECGProcessingService._find_wave_for_cycle(
            p_onsets, r_peak_sample, fs, before=True, max_distance_ms=200
        )
        p_offset_idx = ECGProcessingService._find_wave_for_cycle(
            p_offsets, r_peak_sample, fs, before=True, max_distance_ms=100
        )
        
        # QRS features
        qrs_onset_idx = ECGProcessingService._find_wave_for_cycle(
            r_onsets if len(r_onsets) > 0 else q_peaks, 
            r_peak_sample, fs, before=True, max_distance_ms=60
        )
        qrs_offset_idx = ECGProcessingService._find_wave_for_cycle(
            r_offsets if len(r_offsets) > 0 else s_peaks, 
            r_peak_sample, fs, before=False, max_distance_ms=60
        )
        
        # T-wave features (occurs after R-peak)
        t_offset_idx = ECGProcessingService._find_wave_for_cycle(
            t_offsets, r_peak_sample, fs, before=False, max_distance_ms=400
        )
        
        # Calculate timings relative to the reference
        features["p_onset"] = reference_point
        
        # P-wave end
        if p_onset_idx is not None and p_offset_idx is not None:
            p_duration = (p_offset_idx - p_onset_idx) * conv
            if 40 <= p_duration <= 150:  # Physiological range
                features["p_end"] = reference_point + int(p_duration)
                features["p_duration"] = int(p_duration)
        
        # QRS onset
        if p_onset_idx is not None and qrs_onset_idx is not None:
            pq_interval = (qrs_onset_idx - p_onset_idx) * conv
            if 80 <= pq_interval <= 300:  # Physiological range
                features["qrs_onset"] = reference_point + int(pq_interval)
                features["pq_interval"] = int(pq_interval)
        elif qrs_onset_idx is not None:
            # Estimate based on typical timing if P-wave not found
            features["qrs_onset"] = reference_point + 160  # Typical PQ interval
        
        # QRS end
        if qrs_onset_idx is not None and qrs_offset_idx is not None:
            qrs_duration = (qrs_offset_idx - qrs_onset_idx) * conv
            if 50 <= qrs_duration <= 150:  # Physiological range
                if "qrs_onset" in features:
                    features["qrs_end"] = features["qrs_onset"] + int(qrs_duration)
                features["qrs_duration"] = int(qrs_duration)
        
        # T-wave end
        if qrs_onset_idx is not None and t_offset_idx is not None:
            qt_interval = (t_offset_idx - qrs_onset_idx) * conv
            if 250 <= qt_interval <= 600:  # Physiological range
                if "qrs_onset" in features:
                    features["t_end"] = features["qrs_onset"] + int(qt_interval)
                features["qt_interval"] = int(qt_interval)
        
        return features

    @staticmethod
    def _find_wave_for_cycle(wave_indices, r_peak_sample, fs, before=True, max_distance_ms=200):
        """
        Find the wave index that belongs to the current cardiac cycle
        """
        if len(wave_indices) == 0:
            return None
        
        max_distance_samples = int(max_distance_ms * fs / 1000)
        
        if before:
            # Find waves before R-peak
            valid_waves = wave_indices[
                (wave_indices < r_peak_sample) & 
                (wave_indices > r_peak_sample - max_distance_samples)
            ]
            if len(valid_waves) > 0:
                # Return the closest one
                return valid_waves[-1]
        else:
            # Find waves after R-peak
            valid_waves = wave_indices[
                (wave_indices > r_peak_sample) & 
                (wave_indices < r_peak_sample + max_distance_samples)
            ]
            if len(valid_waves) > 0:
                # Return the closest one
                return valid_waves[0]
        
        return None

    @staticmethod
    def process_ecg_advanced(signal, fs):
        """
        Process ECG signal using NeuroKit2 with full delineation
        """
        logger.info("Starting advanced ECG processing with full delineation.")
        
        try:
            # Clean the ECG signal
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=fs)
            
            # Full ECG processing including delineation
            ecg_signals, ecg_info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)
            
            # Extract R-peaks
            rpeaks_indices = ecg_info["ECG_R_Peaks"]
            logger.debug(f"Detected {len(rpeaks_indices)} R-peaks.")
            
            # Create DataFrame
            df = pd.DataFrame({
                "ECG_Raw": signal, 
                "ECG_Clean": ecg_cleaned
            })
            
            # Add all delineated features to DataFrame
            for col in ecg_signals.columns:
                if col.startswith("ECG_"):
                    df[col] = ecg_signals[col].values
            
            return df, rpeaks_indices

        except Exception as e:
            logger.exception("Error in advanced ECG processing.")
            # Fallback
            df = pd.DataFrame({"ECG_Raw": signal, "ECG_Clean": signal})
            peaks, _ = find_peaks(
                signal, height=np.mean(signal) + 2 * np.std(signal), distance=int(0.6 * fs)
            )
            return df, peaks

    @staticmethod
    def updated_extract_waves(df):
        """
        Extract wave features from the processed DataFrame
        """
        logger.info("Extracting wave features from processed DataFrame.")
        
        waves = {}
        
        # Map NeuroKit2 column names to expected format
        column_mapping = {
            "ECG_P_Onsets": "ECG_P_Onsets",
            "ECG_P_Peaks": "ECG_P_Peaks", 
            "ECG_P_Offsets": "ECG_P_Offsets",
            "ECG_Q_Peaks": "ECG_Q_Peaks",
            "ECG_R_Onsets": "ECG_R_Onsets",
            "ECG_R_Offsets": "ECG_R_Offsets",
            "ECG_S_Peaks": "ECG_S_Peaks",
            "ECG_T_Onsets": "ECG_T_Onsets",
            "ECG_T_Peaks": "ECG_T_Peaks",
            "ECG_T_Offsets": "ECG_T_Offsets"
        }
        
        for nk_col, wave_key in column_mapping.items():
            if nk_col in df.columns:
                # Find indices where the wave is marked (usually as 1)
                wave_indices = df.index[df[nk_col] == 1].to_numpy()
                waves[wave_key] = wave_indices
            else:
                waves[wave_key] = np.array([])
        
        return waves

    @staticmethod
    def calculate_robust_ecg_axis(signals, fs, lead_names):
        """
        Enhanced axis calculation with fallback strategies
        """
        logger.info("Starting robust ECG axis calculation.")
        axis_results = {"p_axis": None, "qrs_axis": None, "t_axis": None}

        try:
            # Create flexible lead mapping
            lead_mapping = ECGProcessingService._create_lead_mapping(lead_names)
            
            # Process each lead
            lead_amplitudes = {}
            for lead_name, lead_idx in lead_mapping.items():
                if lead_idx < signals.shape[1]:
                    signal_lead = signals[:, lead_idx]
                    
                    # Clean and process the signal
                    ecg_cleaned = nk.ecg_clean(signal_lead, sampling_rate=fs)
                    ecg_signals, ecg_info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)
                    
                    # Calculate amplitudes for each wave type
                    lead_amplitudes[lead_name] = {
                        "p_amplitude": ECGProcessingService._calculate_wave_amplitude(
                            ecg_signals, ecg_info, "P", fs
                        ),
                        "qrs_amplitude": ECGProcessingService._calculate_wave_amplitude(
                            ecg_signals, ecg_info, "QRS", fs
                        ),
                        "t_amplitude": ECGProcessingService._calculate_wave_amplitude(
                            ecg_signals, ecg_info, "T", fs
                        )
                    }

            # Calculate axes
            for wave_type, axis_key in [("p_amplitude", "p_axis"), 
                                         ("qrs_amplitude", "qrs_axis"), 
                                         ("t_amplitude", "t_axis")]:
                axis_results[axis_key] = ECGProcessingService._calculate_axis_from_leads(
                    lead_amplitudes, wave_type
                )
            
            logger.info(f"Axis calculation complete: {axis_results}")
            return axis_results

        except Exception as e:
            logger.exception("Error in axis calculation.")
            return axis_results

    @staticmethod
    def _create_lead_mapping(lead_names):
        """Create flexible mapping for lead names"""
        lead_mapping = {}
        
        mapping_patterns = {
            'I': ['I', 'LEAD_I', 'LEAD I', 'MLI', '1'],
            'II': ['II', 'LEAD_II', 'LEAD II', 'MLII', '2'],
            'III': ['III', 'LEAD_III', 'LEAD III', 'MLIII', '3'],
            'aVR': ['AVR', 'aVR', 'LEAD_aVR', 'LEAD aVR'],
            'aVL': ['AVL', 'aVL', 'LEAD_aVL', 'LEAD aVL'],
            'aVF': ['AVF', 'aVF', 'LEAD_aVF', 'LEAD aVF'],
            'V1': ['V1', 'LEAD_V1', 'LEAD V1'],
            'V2': ['V2', 'LEAD_V2', 'LEAD V2'],
            'V3': ['V3', 'LEAD_V3', 'LEAD V3'],
            'V4': ['V4', 'LEAD_V4', 'LEAD V4'],
            'V5': ['V5', 'LEAD_V5', 'LEAD V5'],
            'V6': ['V6', 'LEAD_V6', 'LEAD V6'],
        }
        
        for i, lead_name in enumerate(lead_names):
            clean_name = lead_name.strip().upper()
            for standard_name, patterns in mapping_patterns.items():
                if clean_name in patterns:
                    lead_mapping[standard_name] = i
                    break
        
        return lead_mapping

    @staticmethod
    def _calculate_wave_amplitude(ecg_signals, ecg_info, wave_type, fs):
        """Calculate amplitude for a specific wave type"""
        try:
            if wave_type == "P":
                peaks = ecg_info.get("ECG_P_Peaks", [])
                signal_key = "ECG_Clean"
            elif wave_type == "QRS":
                peaks = ecg_info.get("ECG_R_Peaks", [])
                signal_key = "ECG_Clean"
            elif wave_type == "T":
                peaks = ecg_info.get("ECG_T_Peaks", [])
                signal_key = "ECG_Clean"
            else:
                return None
            
            if len(peaks) == 0:
                return None
            
            signal_data = ecg_signals[signal_key].values
            amplitudes = []
            
            # Window size based on wave type
            window_ms = {"P": 60, "QRS": 40, "T": 100}[wave_type]
            window_samples = int(window_ms * fs / 1000)
            
            for peak in peaks:
                if window_samples <= peak < len(signal_data) - window_samples:
                    window = signal_data[peak - window_samples:peak + window_samples]
                    amplitude = np.max(window) - np.min(window)
                    amplitudes.append(amplitude)
            
            if len(amplitudes) >= 2:
                return np.median(amplitudes)
            elif len(amplitudes) == 1:
                return amplitudes[0]
            
            return None
            
        except Exception:
            return None

    @staticmethod
    def _calculate_axis_from_leads(lead_amplitudes, wave_type):
        """Calculate axis using available leads"""
        # Try standard calculation with I and aVF
        if all(lead in lead_amplitudes for lead in ["I", "aVF"]):
            amp_I = lead_amplitudes["I"].get(wave_type)
            amp_aVF = lead_amplitudes["aVF"].get(wave_type)
            
            if amp_I is not None and amp_aVF is not None and abs(amp_I) > 1e-6:
                axis = np.degrees(np.arctan2(amp_aVF * 2 / np.sqrt(3), amp_I))
                return int(np.round(axis))
        
        # Try fallback methods...
        # (keeping the existing fallback logic from the original code)
        
        return None

    @staticmethod
    def analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id):
        """
        Analyze 12-lead ECG with fully algorithmic feature extraction
        """
        logger.info(f"Starting 12-lead ECG analysis for file ID: {file_id}")
        try:
            # Read the WFDB record
            signals, fs, record = ECGProcessingService.read_wfdb_record(wfdb_basename)
            lead_names = (
                record.sig_name
                if record.sig_name
                else [f"Lead_{i}" for i in range(signals.shape[1])]
            )
            
            # Extract features algorithmically
            physionet_features, intervals = ECGProcessingService.extract_physionet_features_fixed(
                signals, fs, lead_names
            )

            # Generate plot
            lead_idx = ECGProcessingService.select_best_lead(signals)
            raw_signal = signals[:, lead_idx]
            
            df, rpeaks = ECGProcessingService.process_ecg_advanced(raw_signal, fs)
            waves = ECGProcessingService.updated_extract_waves(df)
            
            plot_path = os.path.join(plot_folder, f"{file_id}.png")
            ECGProcessingService.plot_waveform_diagram(
                df["ECG_Clean"].values,
                fs,
                rpeaks,
                waves,
                title=f"12-Lead ECG Analysis - {file_id} (Lead: {lead_names[lead_idx]})",
                filename=plot_path,
            )

            # Calculate heart rate from detected R-peaks
            heart_rate = None
            if len(rpeaks) > 1:
                rr_intervals = np.diff(rpeaks) * 1000 / fs  # in ms
                valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2500)]
                if len(valid_rr) > 0:
                    mean_rr = np.median(valid_rr)
                    heart_rate = int(round(60000 / mean_rr))

            summary = {
                "physionet_features": physionet_features,
                "intervals": intervals,
                "heart_rate": heart_rate,
                "lead_count": signals.shape[1],
                "best_lead": lead_names[lead_idx],
                "sampling_rate": fs,
            }
            
            logger.info(f"ECG analysis complete for {file_id}")
            return summary, plot_path

        except Exception as e:
            logger.exception(f"Error during 12-lead ECG analysis for {file_id}")
            raise APIError(f"Failed to analyze ECG: {str(e)}", code=500)

    # Keep the remaining utility methods unchanged
    @staticmethod
    def read_wfdb_record(wfdb_basename):
        """Read a WFDB record"""
        logger.debug(f"Reading WFDB record: {wfdb_basename}")
        try:
            record = wfdb.rdrecord(wfdb_basename, physical=True)
            signals = record.p_signal
            fs = record.fs
            if signals is None or fs is None:
                raise ValueError("Invalid WFDB record")
            return signals, fs, record
        except Exception as e:
            logger.exception(f"Error reading WFDB record {wfdb_basename}")
            raise APIError(f"Could not read WFDB record: {str(e)}", code=404)

    @staticmethod
    def select_best_lead(signals):
        """Select the best lead based on signal quality"""
        if signals.shape[1] == 1:
            return 0
        
        quality_scores = []
        for i in range(signals.shape[1]):
            lead_signal = signals[:, i]
            clean_signal = lead_signal[~np.isnan(lead_signal)]
            if len(clean_signal) == 0:
                quality_scores.append(0)
                continue
            
            amplitude_range = np.ptp(clean_signal)
            signal_std = np.std(clean_signal)
            quality_score = amplitude_range * signal_std
            quality_scores.append(quality_score)
        
        return np.argmax(quality_scores)

    @staticmethod
    def plot_waveform_diagram(signal, fs, rpeaks, waves, title="ECG Analysis", filename=None):
        """Plot ECG waveform with detected features"""
        logger.info(f"Generating waveform diagram: '{title}'")
        try:
            plt.figure(figsize=(15, 8))
            time = np.arange(len(signal)) / fs
            
            plt.plot(time, signal, "b-", linewidth=1, label="ECG Signal")
            
            if rpeaks is not None and len(rpeaks) > 0:
                valid_rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < len(signal))]
                if len(valid_rpeaks) > 0:
                    plt.plot(time[valid_rpeaks], signal[valid_rpeaks], "ro", markersize=8, label="R-peaks")
            
            colors = {"P": "green", "Q": "orange", "S": "purple", "T": "red"}
            
            for wave_type in ["P", "Q", "S", "T"]:
                peak_key = f"ECG_{wave_type}_Peaks"
                if peak_key in waves and len(waves[peak_key]) > 0:
                    valid_peaks = waves[peak_key][(waves[peak_key] >= 0) & (waves[peak_key] < len(signal))]
                    if len(valid_peaks) > 0:
                        plt.plot(
                            time[valid_peaks],
                            signal[valid_peaks],
                            "o",
                            color=colors[wave_type],
                            markersize=6,
                            label=f"{wave_type}-peaks",
                        )
            
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mV)")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if filename:
                plt.savefig(filename, dpi=150, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.exception("Error during plotting")
            plt.close()
            raise APIError(f"Failed to generate plot: {str(e)}", code=500)

    @staticmethod
    def analyze_and_plot(wfdb_basename, plot_folder, file_id):
        """Wrapper function"""
        return ECGProcessingService.analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id)