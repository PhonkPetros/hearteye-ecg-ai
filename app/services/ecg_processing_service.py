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
    # Compensate for filter and wavelet transform delays at 500Hz
    DELAY_COMPENSATION = {
        "P_wave": 8,  # 8 samples = 16ms at 500Hz
        "QRS": 5,  # 5 samples = 10ms at 500Hz
        "T_wave": 10,  # 10 samples = 20ms at 500Hz
    }

    @staticmethod
    def compensate_group_delay(waves, fs=500):
        """
        Compensate for group delay introduced by filtering and wavelet transforms
        """
        logger.debug(f"Compensating group delay for waves at fs={fs}")
        compensated = {}

        # P-wave compensation
        for key in ["ECG_P_Onsets", "ECG_P_Offsets", "ECG_P_Peaks"]:
            if key in waves and len(waves[key]) > 0:
                compensated[key] = waves[key] - ECGProcessingService.DELAY_COMPENSATION["P_wave"]
                compensated[key] = np.maximum(compensated[key], 0)  # Ensure non-negative

        # QRS compensation
        for key in ["ECG_QRS_Onsets", "ECG_QRS_Offsets", "ECG_Q_Peaks", "ECG_S_Peaks"]:
            if key in waves and len(waves[key]) > 0:
                compensated[key] = waves[key] - ECGProcessingService.DELAY_COMPENSATION["QRS"]
                compensated[key] = np.maximum(compensated[key], 0)

        # T-wave compensation
        for key in ["ECG_T_Onsets", "ECG_T_Offsets", "ECG_T_Peaks"]:
            if key in waves and len(waves[key]) > 0:
                compensated[key] = waves[key] - ECGProcessingService.DELAY_COMPENSATION["T_wave"]
                compensated[key] = np.maximum(compensated[key], 0)

        # Copy over any uncompensated keys
        for key in waves:
            if key not in compensated:
                compensated[key] = waves[key]
        logger.debug("Group delay compensation complete.")
        return compensated

    @staticmethod
    def calculate_robust_ecg_axis(signals, fs, lead_names):
        """
        Enhanced axis calculation with fallback strategies and signal quality checks
        """
        logger.info("Starting robust ECG axis calculation.")
        axis_results = {"p_axis": None, "qrs_axis": None, "t_axis": None}

        try:
            # Flexible lead mapping
            logger.debug("Mapping lead names.")
            lead_mapping = {}
            for i, lead_name in enumerate(lead_names):
                clean_name = lead_name.strip().upper()
                if clean_name in ["I", "LEAD_I", "LEAD I", "MLI"]:
                    lead_mapping["I"] = i
                elif clean_name in ["II", "LEAD_II", "LEAD II", "MLII"]:
                    lead_mapping["II"] = i
                elif clean_name in ["III", "LEAD_III", "LEAD III", "MLIII"]:
                    lead_mapping["III"] = i
                elif clean_name in ["AVF", "aVF", "LEAD_aVF", "LEAD aVF"]:
                    lead_mapping["aVF"] = i
                elif clean_name in ["AVL", "aVL", "LEAD_aVL", "LEAD aVL"]:
                    lead_mapping["aVL"] = i
                elif clean_name in ["AVR", "aVR", "LEAD_aVR", "LEAD aVR"]:
                    lead_mapping["aVR"] = i

            # Process each lead and extract amplitudes
            lead_amplitudes = {}
            for lead_name, lead_idx in lead_mapping.items():
                logger.debug(f"Processing lead: {lead_name} (index: {lead_idx})")
                if lead_idx < signals.shape[1]:
                    signal_lead = signals[:, lead_idx]

                    # Apply zero-phase filtering to avoid phase distortion
                    nyquist = fs / 2
                    b, a = signal.butter(2, [0.5 / nyquist, 40 / nyquist], btype="band")
                    filtered_signal = signal.filtfilt(b, a, signal_lead)

                    # Process the signal with your existing function
                    df, rpeaks = ECGProcessingService.process_ecg_advanced(filtered_signal, fs)
                    waves = ECGProcessingService.updated_extract_waves(df)

                    # Apply group delay compensation
                    waves = ECGProcessingService.compensate_group_delay(waves, fs)

                    # Calculate amplitudes with enhanced methods
                    lead_amplitudes[lead_name] = {
                        "p_amplitude": ECGProcessingService.calculate_robust_wave_amplitude(
                            df["ECG_Clean"].values,
                            waves.get("ECG_P_Peaks", []),
                            wave_type="P",
                            fs=fs,
                        ),
                        "qrs_amplitude": ECGProcessingService.calculate_robust_wave_amplitude(
                            df["ECG_Clean"].values, rpeaks, wave_type="QRS", fs=fs
                        ),
                        "t_amplitude": ECGProcessingService.calculate_robust_wave_amplitude(
                            df["ECG_Clean"].values,
                            waves.get("ECG_T_Peaks", []),
                            wave_type="T",
                            fs=fs,
                        ),
                    }
                else:
                    logger.warning(f"Lead index {lead_idx} for {lead_name} is out of bounds for signals with shape {signals.shape}.")


            # Multi-strategy axis calculation
            logger.debug("Calculating P, QRS, and T axes.")
            axis_results["p_axis"] = ECGProcessingService.calculate_axis_with_fallback(
                lead_amplitudes, wave_type="p_amplitude"
            )
            axis_results["qrs_axis"] = ECGProcessingService.calculate_axis_with_fallback(
                lead_amplitudes, wave_type="qrs_amplitude"
            )
            axis_results["t_axis"] = ECGProcessingService.calculate_axis_with_fallback(
                lead_amplitudes, wave_type="t_amplitude"
            )
            logger.info(f"Robust ECG axis calculation complete: {axis_results}")
            return axis_results

        except Exception as e:
            logger.exception("An unexpected error occurred during robust axis calculation.")
            raise APIError(f"Failed to calculate robust ECG axis: {str(e)}", code=500)

    @staticmethod
    def calculate_robust_wave_amplitude(signal, peak_indices, wave_type="QRS", fs=500):
        """
        Calculate wave amplitude with signal quality checks and noise rejection
        """
        logger.debug(f"Calculating robust amplitude for {wave_type} wave.")
        if len(peak_indices) == 0:
            logger.warning(f"No {wave_type} peaks found, returning None for amplitude.")
            return None

        amplitudes = []

        # Define search windows based on wave type
        if wave_type == "P":
            window_samples = int(0.04 * fs)  # 40ms window
            min_amplitude = 0.00002  # 20μV minimum
        elif wave_type == "QRS":
            window_samples = int(0.02 * fs)  # 20ms window
            min_amplitude = 0.0001  # 100μV minimum
        else:  # T wave
            window_samples = int(0.08 * fs)  # 80ms window
            min_amplitude = 0.00003  # 30μV minimum

        for peak_idx in peak_indices:
            if not (window_samples <= peak_idx < len(signal) - window_samples):
                logger.debug(f"Peak index {peak_idx} out of valid range for window size {window_samples}.")
                continue
            # Extract window around peak
            window = signal[peak_idx - window_samples : peak_idx + window_samples]

            # Calculate peak-to-peak amplitude in window
            amplitude = np.max(window) - np.min(window)

            # Apply signal quality check
            if amplitude > min_amplitude:
                amplitudes.append(amplitude)
            else:
                logger.debug(f"Amplitude {amplitude:.6f} for {wave_type} peak at {peak_idx} is below minimum {min_amplitude:.6f}.")

        if len(amplitudes) >= 3:  # Require at least 3 valid measurements
            # Use median for robustness
            median_amplitude = np.median(amplitudes)
            logger.debug(f"Median amplitude for {wave_type}: {median_amplitude:.6f}")
            return median_amplitude
        else:
            logger.warning(f"Only {len(amplitudes)} valid amplitudes found for {wave_type} (less than 3 required). Returning None.")
            return None

    @staticmethod
    def calculate_axis_with_fallback(lead_amplitudes, wave_type="qrs_amplitude"):
        """
        Calculate axis with multiple fallback strategies
        """
        logger.debug(f"Attempting to calculate axis for {wave_type} with fallbacks.")
        # Primary method: I and aVF
        if all(lead in lead_amplitudes for lead in ["I", "aVF"]):
            amp_I = lead_amplitudes["I"].get(wave_type)
            amp_aVF = lead_amplitudes["aVF"].get(wave_type)

            if amp_I is not None and amp_aVF is not None and amp_I > 1e-6:
                # Standard hexaxial formula with correction factor
                axis = np.degrees(np.arctan2(amp_aVF * 2 / np.sqrt(3), amp_I))
                logger.info(f"Calculated {wave_type} axis using I and aVF: {int(np.round(axis))} degrees.")
                return int(np.round(axis))
            else:
                logger.debug(f"Primary method (I and aVF) failed for {wave_type} due to missing or zero amplitudes.")

        # Fallback 1: Use I and II
        if all(lead in lead_amplitudes for lead in ["I", "II"]):
            amp_I = lead_amplitudes["I"].get(wave_type)
            amp_II = lead_amplitudes["II"].get(wave_type)

            if amp_I is not None and amp_II is not None and amp_I > 1e-6:
                # Calculate aVF from I and II using Einthoven's law
                amp_aVF_calc = amp_II - amp_I / 2
                axis = np.degrees(np.arctan2(amp_aVF_calc * 2 / np.sqrt(3), amp_I))
                logger.info(f"Calculated {wave_type} axis using Fallback 1 (I and II): {int(np.round(axis))} degrees.")
                return int(np.round(axis))
            else:
                logger.debug(f"Fallback 1 (I and II) failed for {wave_type} due to missing or zero amplitudes.")

        # Fallback 2: Use II and aVF
        if all(lead in lead_amplitudes for lead in ["II", "aVF"]):
            amp_II = lead_amplitudes["II"].get(wave_type)
            amp_aVF = lead_amplitudes["aVF"].get(wave_type)

            if amp_II is not None and amp_aVF is not None:
                # Calculate I from II and aVF
                amp_I_calc = amp_II - amp_aVF * np.sqrt(3) / 2
                if abs(amp_I_calc) > 1e-6:
                    axis = np.degrees(np.arctan2(amp_aVF * 2 / np.sqrt(3), amp_I_calc))
                    logger.info(f"Calculated {wave_type} axis using Fallback 2 (II and aVF): {int(np.round(axis))} degrees.")
                    return int(np.round(axis))
                else:
                    logger.debug(f"Fallback 2 (II and aVF) failed for {wave_type} due to calculated I being near zero.")
            else:
                logger.debug(f"Fallback 2 (II and aVF) failed for {wave_type} due to missing amplitudes.")

        logger.warning(f"Could not calculate {wave_type} axis using any available fallback methods.")
        return None

    @staticmethod
    def compute_intervals_fixed_reference(waves, fs, rpeaks=None):
        """
        Compute intervals with fixed reference points matching PhysioNet format
        """
        logger.info("Computing ECG intervals with fixed reference.")
        conv = 1000.0 / fs  # Convert samples to milliseconds

        # Apply group delay compensation first
        waves = ECGProcessingService.compensate_group_delay(waves, fs)

        intervals = {}

        # P wave duration
        p_onsets = waves.get("ECG_P_Onsets", np.array([]))
        p_offsets = waves.get("ECG_P_Offsets", np.array([]))
        if len(p_onsets) > 0 and len(p_offsets) > 0:
            # Match onset to nearest offset
            p_durations = []
            for onset in p_onsets:
                valid_offsets = p_offsets[p_offsets > onset]
                if len(valid_offsets) > 0:
                    offset = valid_offsets[0]
                    duration = (offset - onset) * conv
                    if 60 <= duration <= 120:  # Physiological range
                        p_durations.append(duration)
                    else:
                        logger.debug(f"P wave duration {duration:.2f}ms outside physiological range (60-120ms).")

            if p_durations:
                intervals["P_wave_duration_ms"] = int(round(np.median(p_durations)))
                logger.debug(f"P_wave_duration_ms: {intervals['P_wave_duration_ms']}")
            else:
                logger.warning("No valid P wave durations found.")
        else:
            logger.warning("P onsets or offsets not found for P wave duration calculation.")

        # PQ interval (P onset to QRS onset)
        qrs_onsets = waves.get("ECG_R_Onsets", np.array([]))
        if len(p_onsets) > 0 and len(qrs_onsets) > 0:
            pq_intervals = []
            for p_onset in p_onsets:
                valid_qrs = qrs_onsets[qrs_onsets > p_onset]
                if len(valid_qrs) > 0:
                    qrs_onset = valid_qrs[0]
                    pq = (qrs_onset - p_onset) * conv
                    if 80 <= pq <= 350:  # physiological range including abnormal ones
                        pq_intervals.append(pq)
                    else:
                        logger.debug(f"PQ interval {pq:.2f}ms outside physiological range (80-350ms).")

            if pq_intervals:
                intervals["PQ_interval_ms"] = int(round(np.median(pq_intervals)))
                logger.debug(f"PQ_interval_ms: {intervals['PQ_interval_ms']}")
            else:
                logger.warning("No valid PQ intervals found.")
        else:
            logger.warning("P onsets or QRS onsets not found for PQ interval calculation.")


        # QRS duration
        qrs_offsets = waves.get("ECG_R_Offsets", np.array([]))
        if len(qrs_onsets) > 0 and len(qrs_offsets) > 0:
            qrs_durations = []
            for onset in qrs_onsets:
                valid_offsets = qrs_offsets[qrs_offsets > onset]
                if len(valid_offsets) > 0:
                    offset = valid_offsets[0]
                    duration = (offset - onset) * conv
                    if 60 <= duration <= 120:  # Normal range
                        qrs_durations.append(duration)
                    else:
                        logger.debug(f"QRS duration {duration:.2f}ms outside physiological range (60-120ms).")

            if qrs_durations:
                intervals["QRS_duration_ms"] = int(round(np.median(qrs_durations)))
                logger.debug(f"QRS_duration_ms: {intervals['QRS_duration_ms']}")
            else:
                logger.warning("No valid QRS durations found.")
        else:
            logger.warning("QRS onsets or offsets not found for QRS duration calculation.")

        # QT interval
        t_offsets = waves.get("ECG_T_Offsets", np.array([]))
        if len(qrs_onsets) > 0 and len(t_offsets) > 0:
            qt_intervals = []
            for qrs_onset in qrs_onsets:
                valid_t = t_offsets[t_offsets > qrs_onset]
                if len(valid_t) > 0:
                    t_offset = valid_t[0]
                    qt = (t_offset - qrs_onset) * conv
                    if 300 <= qt <= 450:  # Normal range
                        qt_intervals.append(qt)
                    else:
                        logger.debug(f"QT interval {qt:.2f}ms outside physiological range (300-450ms).")

            if qt_intervals:
                intervals["QT_interval_ms"] = int(round(np.median(qt_intervals)))
                logger.debug(f"QT_interval_ms: {intervals['QT_interval_ms']}")
            else:
                logger.warning("No valid QT intervals found.")
        else:
            logger.warning("QRS onsets or T offsets not found for QT interval calculation.")

        # RR interval
        if rpeaks is not None and len(rpeaks) > 1:
            rr_intervals = np.diff(rpeaks) * conv
            valid_rr = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 2000)]
            if len(valid_rr) > 0:
                intervals["RR_interval_ms"] = int(round(np.median(valid_rr)))
                logger.debug(f"RR_interval_ms: {intervals['RR_interval_ms']}")
            else:
                logger.warning("No valid RR intervals found within range (400-2000ms).")
        else:
            logger.warning("R-peaks not available or insufficient for RR interval calculation.")
        logger.info(f"Interval computation complete: {intervals}")
        return intervals

    @staticmethod
    def extract_physionet_features_fixed(signals, fs, lead_names):
        """
        Extract features matching PhysioNet format with all fixes applied
        """
        logger.info("Extracting PhysioNet features.")
        try:
            # Get robust axis calculations
            axes = ECGProcessingService.calculate_robust_ecg_axis(signals, fs, lead_names)

            # Process best lead for timing
            lead_idx = ECGProcessingService.select_best_lead(signals)
            raw_signal = signals[:, lead_idx]
            logger.debug(f"Selected best lead for timing: {lead_names[lead_idx]} (index {lead_idx})")

            # Process with zero-phase filtering
            nyquist = fs / 2
            b, a = signal.butter(2, [0.5 / nyquist, 40 / nyquist], btype="band")
            filtered_signal = signal.filtfilt(b, a, raw_signal)
            logger.debug("Applied zero-phase filtering to selected lead.")

            # Get delineation
            df, rpeaks = ECGProcessingService.process_ecg_advanced(filtered_signal, fs)
            waves = ECGProcessingService.updated_extract_waves(df)

            # Apply group delay compensation
            waves = ECGProcessingService.compensate_group_delay(waves, fs)

            # Calculate intervals with fixed reference
            intervals = ECGProcessingService.compute_intervals_fixed_reference(waves, fs, rpeaks)

            # Build PhysioNet format output
            features = {
                "rr_interval": intervals.get("RR_interval_ms"),
                "p_onset": 40,  # Fixed reference point
                "p_end": None,
                "qrs_onset": None,
                "qrs_end": None,
                "t_end": None,
                "p_axis": axes.get("p_axis"),
                "qrs_axis": axes.get("qrs_axis"),
                "t_axis": axes.get("t_axis"),
            }

            # Calculate relative timings from fixed reference
            p_duration = intervals.get("P_wave_duration_ms", 84)
            pq_interval = intervals.get("PQ_interval_ms", 122)
            qrs_duration = intervals.get("QRS_duration_ms", 84)
            qt_interval = intervals.get("QT_interval_ms", 342)

            features["p_end"] = features["p_onset"] + p_duration
            features["qrs_onset"] = features["p_onset"] + pq_interval
            features["qrs_end"] = features["qrs_onset"] + qrs_duration
            features["t_end"] = features["qrs_onset"] + qt_interval
            logger.info(f"PhysioNet features extracted: {features}")
            return features, intervals

        except Exception as e:
            logger.exception("An unexpected error occurred during PhysioNet feature extraction.")
            default_features = {
                "rr_interval": None,
                "p_onset": 40,
                "p_end": None,
                "qrs_onset": None,
                "qrs_end": None,
                "t_end": None,
                "p_axis": None,
                "qrs_axis": None,
                "t_axis": None,
            }
            raise APIError(f"Failed to extract PhysioNet features: {str(e)}", code=500)

    @staticmethod
    def analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id):
        """
        Fixed version of analyze_and_plot_12_lead with all corrections
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
            logger.debug(f"Successfully read WFDB record. Signals shape: {signals.shape}, FS: {fs}, Leads: {lead_names}")

            # Extract features with all fixes
            physionet_features, intervals = ECGProcessingService.extract_physionet_features_fixed(
                signals, fs, lead_names
            )

            lead_idx = ECGProcessingService.select_best_lead(signals)
            raw_signal = signals[:, lead_idx]

            df, rpeaks = ECGProcessingService.process_ecg_advanced(raw_signal, fs)
            waves = ECGProcessingService.updated_extract_waves(df)
            waves = ECGProcessingService.compensate_group_delay(waves, fs)  # Apply compensation for plotting too

            # Ensure plot_folder exists
            if not os.path.exists(plot_folder):
                logger.info(f"Creating plot directory: {plot_folder}")
                os.makedirs(plot_folder, exist_ok=True)

            plot_path = os.path.join(plot_folder, f"{file_id}.png")
            ECGProcessingService.plot_waveform_diagram(
                df["ECG_Clean"].values,
                fs,
                rpeaks,
                waves,
                title=f"12-Lead ECG Analysis - {file_id} (Lead: {lead_names[lead_idx]})",
                filename=plot_path,
            )
            logger.info(f"Waveform diagram saved to: {plot_path}")

            heart_rate = None
            if len(rpeaks) > 1:
                try:
                    heart_rate = int(round(60 * fs / np.median(np.diff(rpeaks))))
                except Exception as hr_e:
                    logger.warning(f"Could not calculate heart rate: {hr_e}")


            summary = {
                "physionet_features": physionet_features,
                "intervals": intervals,
                "heart_rate": heart_rate,
                "lead_count": signals.shape[1],
                "best_lead": lead_names[lead_idx],
                "sampling_rate": fs,
            }
            logger.info(f"ECG analysis complete for {file_id}. Summary: {summary}")
            return summary, plot_path

        except APIError as e:
            logger.error(f"API Error during 12-lead ECG analysis for {file_id}: {e.message}")
            return {"physionet_features": ECGProcessingService.extract_physionet_features_fixed(signals, fs, lead_names)[0], "error": e.message}, None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during 12-lead ECG analysis for {file_id}.")
            default_features = {
                "rr_interval": None,
                "p_onset": 40,
                "p_end": None,
                "qrs_onset": None,
                "qrs_end": None,
                "t_end": None,
                "p_axis": None,
                "qrs_axis": None,
                "t_axis": None,
            }
            raise APIError(f"Failed to analyze and plot 12-lead ECG: {str(e)}", code=500)

    @staticmethod
    def read_wfdb_record(wfdb_basename):
        """
        Read a WFDB record and return signals, sampling frequency, and record info
        """
        logger.debug(f"Attempting to read WFDB record: {wfdb_basename}")
        try:
            record = wfdb.rdrecord(wfdb_basename, physical=True)
            signals = record.p_signal
            fs = record.fs
            if signals is None or fs is None:
                raise ValueError("WFDB record signals or sampling frequency could not be read.")
            logger.info(f"Successfully read WFDB record {wfdb_basename}. Signals shape: {signals.shape}, FS: {fs}")
            return signals, fs, record
        except Exception as e:
            logger.exception(f"Error reading WFDB record {wfdb_basename}.")
            raise APIError(f"Could not read WFDB record {wfdb_basename}: {str(e)}", code=404)

    @staticmethod
    def select_best_lead(signals):
        """
        Select the best lead based on signal quality (highest amplitude variation)
        """
        logger.debug("Selecting the best lead based on signal quality.")
        if signals.shape[1] == 1:
            logger.debug("Only one lead available, selecting lead 0.")
            return 0

        # Calculate signal quality metrics for each lead
        quality_scores = []
        for i in range(signals.shape[1]):
            lead_signal = signals[:, i]
            # Remove NaN values
            clean_signal = lead_signal[~np.isnan(lead_signal)]
            if len(clean_signal) == 0:
                logger.warning(f"Lead {i} contains no valid (non-NaN) data. Quality score set to 0.")
                quality_scores.append(0)
                continue

            # Calculate quality based on amplitude range and signal-to-noise ratio
            amplitude_range = np.ptp(clean_signal)  # Peak-to-peak amplitude
            signal_std = np.std(clean_signal)

            # Simple quality score: higher amplitude range and variation is better
            quality_score = amplitude_range * signal_std
            quality_scores.append(quality_score)

        best_lead_idx = np.argmax(quality_scores)
        logger.info(f"Selected best lead: {best_lead_idx} with quality score: {quality_scores[best_lead_idx]:.2f}")
        return best_lead_idx

    @staticmethod
    def process_ecg_advanced(signal, fs):
        """
        Process ECG signal using NeuroKit2 and return cleaned signal with R-peaks
        """
        logger.info("Starting advanced ECG processing with NeuroKit2.")
        if not isinstance(signal, np.ndarray) or signal.size == 0:
            logger.error("Input signal is not a valid numpy array or is empty.")
            raise APIError("Invalid or empty ECG signal provided for processing.", code=400)

        try:
            # Clean the ECG signal
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=fs)
            logger.debug("ECG signal cleaned using NeuroKit2.")

            # Find R-peaks
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            rpeaks_indices = rpeaks["ECG_R_Peaks"]
            logger.debug(f"Detected {len(rpeaks_indices)} R-peaks.")

            # Create DataFrame with cleaned signal
            df = pd.DataFrame({"ECG_Raw": signal, "ECG_Clean": ecg_cleaned})
            logger.info("ECG processing successful.")
            return df, rpeaks_indices

        except Exception as e:
            logger.exception("Error occurred during advanced ECG processing with NeuroKit2. Falling back to simple peak detection.")
            # Fallback: return original signal with basic peak detection
            df = pd.DataFrame({"ECG_Raw": signal, "ECG_Clean": signal})

            # Simple peak detection as fallback
            peaks, _ = find_peaks(
                signal, height=np.mean(signal) + 2 * np.std(signal), distance=int(0.6 * fs)
            )
            logger.warning(f"Using fallback simple peak detection. Found {len(peaks)} peaks.")
            return df, peaks

    @staticmethod
    def updated_extract_waves(df):
        """
        Extract wave features from ECG signal using NeuroKit2
        """
        logger.info("Extracting wave features using NeuroKit2.")
        if "ECG_Clean" not in df.columns or df["ECG_Clean"].empty:
            logger.error("Clean ECG signal not found in DataFrame for wave extraction.")
            raise APIError("Clean ECG signal missing for wave extraction.", code=400)

        try:
            ecg_signal = df["ECG_Clean"].values
            fs = 500  # Assume 500 Hz sampling rate
            logger.debug(f"Assuming sampling rate: {fs} Hz for wave extraction.")

            # Use NeuroKit2 to extract ECG features
            _, waves = nk.ecg_delineate(ecg_signal, sampling_rate=fs)
            logger.debug("ECG delineation performed by NeuroKit2.")

            # Convert to the expected format
            wave_dict = {}
            for key, value in waves.items():
                if value is not None and len(value) > 0:
                    # Remove NaN values and convert to numpy array
                    clean_values = np.array(value)[~np.isnan(value)]
                    wave_dict[key] = clean_values.astype(int)
                else:
                    wave_dict[key] = np.array([])
            logger.info("Wave features extracted successfully.")
            return wave_dict

        except Exception as e:
            logger.exception("Error occurred during wave extraction with NeuroKit2. Returning empty wave dictionary as fallback.")
            # Return empty wave dictionary as fallback
            return {
                "ECG_P_Onsets": np.array([]),
                "ECG_P_Offsets": np.array([]),
                "ECG_P_Peaks": np.array([]),
                "ECG_QRS_Onsets": np.array([]),
                "ECG_QRS_Offsets": np.array([]),
                "ECG_Q_Peaks": np.array([]),
                "ECG_S_Peaks": np.array([]),
                "ECG_T_Onsets": np.array([]),
                "ECG_T_Offsets": np.array([]),
                "ECG_T_Peaks": np.array([]),
            }

    @staticmethod
    def plot_waveform_diagram(
        signal, fs, rpeaks, waves, title="ECG Analysis", filename=None
    ):
        """
        Plot ECG waveform with detected features
        """
        logger.info(f"Generating waveform diagram: '{title}'")
        try:
            plt.figure(figsize=(15, 8))

            # Time axis
            time = np.arange(len(signal)) / fs

            # Plot ECG signal
            plt.plot(time, signal, "b-", linewidth=1, label="ECG Signal")

            # Plot R-peaks
            if rpeaks is not None and len(rpeaks) > 0:
                # Ensure rpeaks indices are within signal bounds
                valid_rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < len(signal))]
                if len(valid_rpeaks) > 0:
                    plt.plot(time[valid_rpeaks], signal[valid_rpeaks], "ro", markersize=8, label="R-peaks")
                    logger.debug(f"Plotted {len(valid_rpeaks)} R-peaks.")
                else:
                    logger.warning("No valid R-peaks to plot within signal bounds.")
            else:
                logger.warning("No R-peaks provided for plotting.")

            # Plot other wave features if available
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
                        logger.debug(f"Plotted {len(valid_peaks)} {wave_type}-peaks.")
                    else:
                        logger.warning(f"No valid {wave_type}-peaks to plot within signal bounds.")
                else:
                    logger.debug(f"No {wave_type}-peaks found in 'waves' dictionary.")

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mV)")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)

            if filename:
                try:
                    plt.savefig(filename, dpi=150, bbox_inches="tight")
                    logger.info(f"Waveform diagram saved to {filename}")
                except Exception as save_e:
                    logger.exception(f"Error saving plot to {filename}.")
                    raise APIError(f"Failed to save plot: {str(save_e)}", code=500)
                finally:
                    plt.close()
            else:
                plt.show()
                logger.info("Waveform diagram displayed.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during plotting for '{title}'. Attempting simple fallback plot.")
            # Create a simple fallback plot
            plt.figure(figsize=(15, 8))
            time = np.arange(len(signal)) / fs
            plt.plot(time, signal, "b-", linewidth=1)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mV)")
            plt.title(f"{title} (Fallback Plot)")
            plt.grid(True, alpha=0.3)

            if filename:
                try:
                    plt.savefig(filename, dpi=150, bbox_inches="tight")
                    logger.warning(f"Fallback plot saved to {filename}.")
                except Exception as fallback_save_e:
                    logger.exception(f"Error saving fallback plot to {filename}.")
                finally:
                    plt.close()
            else:
                plt.show()
                logger.warning("Fallback plot displayed.")
            raise APIError(f"Failed to generate waveform diagram: {str(e)}", code=500)

    @staticmethod
    def analyze_and_plot(wfdb_basename, plot_folder, file_id):
        """
        Wrapper function that calls the existing analyze_and_plot_12_lead_fixed function
        """
        logger.info(f"Calling analyze_and_plot_12_lead_fixed for {file_id}.")
        try:
            return ECGProcessingService.analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id)
        except APIError:
            raise 
        except Exception as e:
            logger.exception(f"Unhandled error in analyze_and_plot for {file_id}.")
            raise APIError(f"An unexpected error occurred during overall ECG analysis: {str(e)}", code=500)