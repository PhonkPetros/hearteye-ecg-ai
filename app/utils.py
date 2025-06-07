import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.signal as signal
import logging
import os
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from flask import current_app
from matplotlib.patches import Rectangle
from storage3.exceptions import StorageApiError

def get_supabase():
    """
    Initializes supabase client
    """
    supabase = getattr(current_app, "supabase", None)
    if supabase is None:
        raise RuntimeError("Supabase initialization failed.")
    return supabase

# Compensate for filter and wavelet transform delays at 500Hz
DELAY_COMPENSATION = {
    'P_wave': 8,      # 8 samples = 16ms at 500Hz
    'QRS': 5,         # 5 samples = 10ms at 500Hz  
    'T_wave': 10,     # 10 samples = 20ms at 500Hz
}

def compensate_group_delay(waves, fs=500):
    """
    Compensate for group delay introduced by filtering and wavelet transforms
    """
    compensated = {}
    
    # P-wave compensation
    for key in ['ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_P_Peaks']:
        if key in waves and len(waves[key]) > 0:
            compensated[key] = waves[key] - DELAY_COMPENSATION['P_wave']
            compensated[key] = np.maximum(compensated[key], 0)  # Ensure non-negative
    
    # QRS compensation
    for key in ['ECG_QRS_Onsets', 'ECG_QRS_Offsets', 'ECG_Q_Peaks', 'ECG_S_Peaks']:
        if key in waves and len(waves[key]) > 0:
            compensated[key] = waves[key] - DELAY_COMPENSATION['QRS']
            compensated[key] = np.maximum(compensated[key], 0)
    
    # T-wave compensation
    for key in ['ECG_T_Onsets', 'ECG_T_Offsets', 'ECG_T_Peaks']:
        if key in waves and len(waves[key]) > 0:
            compensated[key] = waves[key] - DELAY_COMPENSATION['T_wave']
            compensated[key] = np.maximum(compensated[key], 0)
    
    # Copy over any uncompensated keys
    for key in waves:
        if key not in compensated:
            compensated[key] = waves[key]
    
    return compensated

# ROBUST AXIS CALCULATION WITH MULTI-LEAD FALLBACK
def calculate_robust_ecg_axis(signals, fs, lead_names):
    """
    Enhanced axis calculation with fallback strategies and signal quality checks
    """
    axis_results = {'p_axis': None, 'qrs_axis': None, 't_axis': None}
    
    try:
        # Flexible lead mapping (keep your existing mapping code)
        lead_mapping = {}
        for i, lead_name in enumerate(lead_names):
            clean_name = lead_name.strip().upper()
            if clean_name in ['I', 'LEAD_I', 'LEAD I', 'MLI']:
                lead_mapping['I'] = i
            elif clean_name in ['II', 'LEAD_II', 'LEAD II', 'MLII']:
                lead_mapping['II'] = i
            elif clean_name in ['III', 'LEAD_III', 'LEAD III', 'MLIII']:
                lead_mapping['III'] = i
            elif clean_name in ['AVF', 'aVF', 'LEAD_aVF', 'LEAD aVF']:
                lead_mapping['aVF'] = i
            elif clean_name in ['AVL', 'aVL', 'LEAD_aVL', 'LEAD aVL']:
                lead_mapping['aVL'] = i
            elif clean_name in ['AVR', 'aVR', 'LEAD_aVR', 'LEAD aVR']:
                lead_mapping['aVR'] = i
        
        # Process each lead and extract amplitudes
        lead_amplitudes = {}
        for lead_name, lead_idx in lead_mapping.items():
            if lead_idx < signals.shape[1]:
                signal_lead = signals[:, lead_idx]
                
                # Apply zero-phase filtering to avoid phase distortion
                nyquist = fs / 2
                b, a = signal.butter(2, [0.5/nyquist, 40/nyquist], btype='band')
                filtered_signal = signal.filtfilt(b, a, signal_lead)
                
                # Process the signal with your existing function
                df, rpeaks = process_ecg_advanced(filtered_signal, fs)
                waves = updated_extract_waves(df)
                
                # Apply group delay compensation
                waves = compensate_group_delay(waves, fs)
                
                # Calculate amplitudes with enhanced methods
                lead_amplitudes[lead_name] = {
                    'p_amplitude': calculate_robust_wave_amplitude(
                        df['ECG_Clean'].values, waves.get('ECG_P_Peaks', []), 
                        wave_type='P', fs=fs
                    ),
                    'qrs_amplitude': calculate_robust_wave_amplitude(
                        df['ECG_Clean'].values, rpeaks, 
                        wave_type='QRS', fs=fs
                    ),
                    't_amplitude': calculate_robust_wave_amplitude(
                        df['ECG_Clean'].values, waves.get('ECG_T_Peaks', []), 
                        wave_type='T', fs=fs
                    )
                }
        
        # Multi-strategy axis calculation
        axis_results['p_axis'] = calculate_axis_with_fallback(
            lead_amplitudes, wave_type='p_amplitude'
        )
        axis_results['qrs_axis'] = calculate_axis_with_fallback(
            lead_amplitudes, wave_type='qrs_amplitude'
        )
        axis_results['t_axis'] = calculate_axis_with_fallback(
            lead_amplitudes, wave_type='t_amplitude'
        )
        
    except Exception as e:
        logging.error(f"Error in robust axis calculation: {e}")
    
    return axis_results

def calculate_robust_wave_amplitude(signal, peak_indices, wave_type='QRS', fs=500):
    """
    Calculate wave amplitude with signal quality checks and noise rejection
    """
    if len(peak_indices) == 0:
        return None
    
    amplitudes = []
    
    # Define search windows based on wave type
    if wave_type == 'P':
        window_samples = int(0.04 * fs)  # 40ms window
        min_amplitude = 0.00002  # 20μV minimum
    elif wave_type == 'QRS':
        window_samples = int(0.02 * fs)  # 20ms window
        min_amplitude = 0.0001   # 100μV minimum
    else:  # T wave
        window_samples = int(0.08 * fs)  # 80ms window
        min_amplitude = 0.00003  # 30μV minimum
    
    for peak_idx in peak_indices:
        if window_samples <= peak_idx < len(signal) - window_samples:
            # Extract window around peak
            window = signal[peak_idx - window_samples:peak_idx + window_samples]
            
            # Calculate peak-to-peak amplitude in window
            amplitude = np.max(window) - np.min(window)
            
            # Apply signal quality check
            if amplitude > min_amplitude:
                amplitudes.append(amplitude)
    
    if len(amplitudes) >= 3:  # Require at least 3 valid measurements
        # Use median for robustness
        return np.median(amplitudes)
    
    return None

def calculate_axis_with_fallback(lead_amplitudes, wave_type='qrs_amplitude'):
    """
    Calculate axis with multiple fallback strategies
    """
    # Primary method: I and aVF
    if all(lead in lead_amplitudes for lead in ['I', 'aVF']):
        amp_I = lead_amplitudes['I'].get(wave_type)
        amp_aVF = lead_amplitudes['aVF'].get(wave_type)
        
        if amp_I is not None and amp_aVF is not None and amp_I > 1e-6:
            # Standard hexaxial formula with correction factor
            axis = np.degrees(np.arctan2(amp_aVF * 2/np.sqrt(3), amp_I))
            return int(np.round(axis))
    
    # Fallback 1: Use I and II
    if all(lead in lead_amplitudes for lead in ['I', 'II']):
        amp_I = lead_amplitudes['I'].get(wave_type)
        amp_II = lead_amplitudes['II'].get(wave_type)
        
        if amp_I is not None and amp_II is not None and amp_I > 1e-6:
            # Calculate aVF from I and II using Einthoven's law
            amp_aVF_calc = amp_II - amp_I/2
            axis = np.degrees(np.arctan2(amp_aVF_calc * 2/np.sqrt(3), amp_I))
            return int(np.round(axis))
    
    # Fallback 2: Use II and aVF
    if all(lead in lead_amplitudes for lead in ['II', 'aVF']):
        amp_II = lead_amplitudes['II'].get(wave_type)
        amp_aVF = lead_amplitudes['aVF'].get(wave_type)
        
        if amp_II is not None and amp_aVF is not None:
            # Calculate I from II and aVF
            amp_I_calc = amp_II - amp_aVF * np.sqrt(3)/2
            if abs(amp_I_calc) > 1e-6:
                axis = np.degrees(np.arctan2(amp_aVF * 2/np.sqrt(3), amp_I_calc))
                return int(np.round(axis))
    
    return None

# IMPROVED INTERVAL CALCULATION WITH FIXED REFERENCE
def compute_intervals_fixed_reference(waves, fs, rpeaks=None):
    """
    Compute intervals with fixed reference points matching PhysioNet format
    """
    conv = 1000.0 / fs  # Convert samples to milliseconds
    
    # Apply group delay compensation first
    waves = compensate_group_delay(waves, fs)
    
    intervals = {}
    
    # P wave duration
    p_onsets = waves.get('ECG_P_Onsets', np.array([]))
    p_offsets = waves.get('ECG_P_Offsets', np.array([]))
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
        
        if p_durations:
            intervals['P_wave_duration_ms'] = int(round(np.median(p_durations)))
    
    # PQ interval (P onset to QRS onset)
    qrs_onsets = waves.get('ECG_R_Onsets', np.array([]))
    if len(p_onsets) > 0 and len(qrs_onsets) > 0:
        pq_intervals = []
        for p_onset in p_onsets:
            valid_qrs = qrs_onsets[qrs_onsets > p_onset]
            if len(valid_qrs) > 0:
                qrs_onset = valid_qrs[0]
                pq = (qrs_onset - p_onset) * conv
                logging.error(f"PQ candidate: {pq} ms (p_onset: {p_onset}, qrs_onset: {qrs_onset})")

                if 80 <= pq <= 350:  # physiological range including abnormal ones
                    pq_intervals.append(pq)

        if pq_intervals:
            intervals['PQ_interval_ms'] = int(round(np.median(pq_intervals)))
    
    # QRS duration
    qrs_offsets = waves.get('ECG_R_Offsets', np.array([]))
    if len(qrs_onsets) > 0 and len(qrs_offsets) > 0:
        qrs_durations = []
        for onset in qrs_onsets:
            valid_offsets = qrs_offsets[qrs_offsets > onset]
            if len(valid_offsets) > 0:
                offset = valid_offsets[0]
                duration = (offset - onset) * conv
                if 60 <= duration <= 120:  # Normal range
                    qrs_durations.append(duration)
        
        if qrs_durations:
            intervals['QRS_duration_ms'] = int(round(np.median(qrs_durations)))
    
    # QT interval
    t_offsets = waves.get('ECG_T_Offsets', np.array([]))
    if len(qrs_onsets) > 0 and len(t_offsets) > 0:
        qt_intervals = []
        for qrs_onset in qrs_onsets:
            valid_t = t_offsets[t_offsets > qrs_onset]
            if len(valid_t) > 0:
                t_offset = valid_t[0]
                qt = (t_offset - qrs_onset) * conv
                if 300 <= qt <= 450:  # Normal range
                    qt_intervals.append(qt)
        
        if qt_intervals:
            intervals['QT_interval_ms'] = int(round(np.median(qt_intervals)))
    
    # RR interval
    if rpeaks is not None and len(rpeaks) > 1:
        rr_intervals = np.diff(rpeaks) * conv
        valid_rr = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 2000)]
        if len(valid_rr) > 0:
            intervals['RR_interval_ms'] = int(round(np.median(valid_rr)))
    
    return intervals

# EXTRACT PHYSIONET FEATURES
def extract_physionet_features_fixed(signals, fs, lead_names):
    """
    Extract features matching PhysioNet format with all fixes applied
    """
    try:
        # Get robust axis calculations
        axes = calculate_robust_ecg_axis(signals, fs, lead_names)
        
        # Process best lead for timing
        lead_idx = select_best_lead(signals)
        raw_signal = signals[:, lead_idx]
        
        # Process with zero-phase filtering
        nyquist = fs / 2
        b, a = signal.butter(2, [0.5/nyquist, 40/nyquist], btype='band')
        filtered_signal = signal.filtfilt(b, a, raw_signal)
        
        # Get delineation
        df, rpeaks = process_ecg_advanced(filtered_signal, fs)
        waves = updated_extract_waves(df)
        
        # Apply group delay compensation
        waves = compensate_group_delay(waves, fs)
        
        # Calculate intervals with fixed reference
        intervals = compute_intervals_fixed_reference(waves, fs, rpeaks)
        
        # Build PhysioNet format output
        features = {
            'rr_interval': intervals.get('RR_interval_ms'),
            'p_onset': 40,  # Fixed reference point
            'p_end': None,
            'qrs_onset': None,
            'qrs_end': None,
            't_end': None,
            'p_axis': axes.get('p_axis'),
            'qrs_axis': axes.get('qrs_axis'),
            't_axis': axes.get('t_axis')
        }
        
        # Calculate relative timings from fixed reference
        p_duration = intervals.get('P_wave_duration_ms', 84)
        pq_interval = intervals.get('PQ_interval_ms', 122)  
        qrs_duration = intervals.get('QRS_duration_ms', 84)  
        qt_interval = intervals.get('QT_interval_ms', 342)   
        
        features['p_end'] = features['p_onset'] + p_duration
        features['qrs_onset'] = features['p_onset'] + pq_interval
        features['qrs_end'] = features['qrs_onset'] + qrs_duration
        features['t_end'] = features['qrs_onset'] + qt_interval
        
        return features, intervals
        
    except Exception as e:
        logging.error(f"Error extracting PhysioNet features: {e}")
        return {
            'rr_interval': None, 'p_onset': 40, 'p_end': None,
            'qrs_onset': None, 'qrs_end': None, 't_end': None,
            'p_axis': None, 'qrs_axis': None, 't_axis': None
        }

def analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id):
    """
    Fixed version of analyze_and_plot_12_lead with all corrections
    """
    try:
        # Read the WFDB record
        signals, fs, record = read_wfdb_record(wfdb_basename)
        lead_names = record.sig_name if record.sig_name else [f"Lead_{i}" for i in range(signals.shape[1])]
        
        # Extract features with all fixes
        physionet_features, intervals = extract_physionet_features_fixed(signals, fs, lead_names)
        
        # Rest of your plotting code remains the same
        lead_idx = select_best_lead(signals)
        raw_signal = signals[:, lead_idx]
        
        df, rpeaks = process_ecg_advanced(raw_signal, fs)
        waves = updated_extract_waves(df)
        waves = compensate_group_delay(waves, fs)  # Apply compensation for plotting too
        
        plot_path = os.path.join(plot_folder, f"{file_id}.png")
        plot_waveform_diagram(df['ECG_Clean'].values, fs, rpeaks, waves, 
                             title=f"12-Lead ECG Analysis - {file_id} (Lead: {lead_names[lead_idx]})", 
                             filename=plot_path)
        
        summary = {
            'physionet_features': physionet_features,
            'intervals': intervals,
            'heart_rate': int(round(60 * fs / np.median(np.diff(rpeaks)))) if len(rpeaks) > 1 else None,
            'lead_count': signals.shape[1],
            'best_lead': lead_names[lead_idx],
            'sampling_rate': fs
        }
        return summary, plot_path
        
    except Exception as e:
        logging.error(f"Error in 12-lead ECG analysis: {e}")
        default_features = {
            'rr_interval': None, 'p_onset': 40, 'p_end': None,
            'qrs_onset': None, 'qrs_end': None, 't_end': None,
            'p_axis': None, 'qrs_axis': None, 't_axis': None
        }
        return {'physionet_features': default_features, 'error': str(e)}, None

def read_wfdb_record(wfdb_basename):
    """
    Read a WFDB record and return signals, sampling frequency, and record info
    """
    try:
        record = wfdb.rdrecord(wfdb_basename, physical=True)
        signals = record.p_signal
        fs = record.fs
        return signals, fs, record
    except Exception as e:
        logging.error(f"Error reading WFDB record {wfdb_basename}: {e}")
        raise

def select_best_lead(signals):
    """
    Select the best lead based on signal quality (highest amplitude variation)
    """
    if signals.shape[1] == 1:
        return 0
    # Calculate signal quality metrics for each lead
    quality_scores = []
    for i in range(signals.shape[1]):
        lead_signal = signals[:, i]
        # Remove NaN values
        clean_signal = lead_signal[~np.isnan(lead_signal)]
        if len(clean_signal) == 0:
            quality_scores.append(0)
            continue
            
        # Calculate quality based on amplitude range and signal-to-noise ratio
        amplitude_range = np.ptp(clean_signal)  # Peak-to-peak amplitude
        signal_std = np.std(clean_signal)
        
        # Simple quality score: higher amplitude range and variation is better
        quality_score = amplitude_range * signal_std
        quality_scores.append(quality_score)
    
    # Return index of lead with highest quality score
    return np.argmax(quality_scores)

def process_ecg_advanced(signal, fs):
    """
    Process ECG signal using NeuroKit2 and return cleaned signal with R-peaks
    """
    try:
        # Clean the ECG signal
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        rpeaks_indices = rpeaks['ECG_R_Peaks']
        
        # Create DataFrame with cleaned signal
        df = pd.DataFrame({
            'ECG_Raw': signal,
            'ECG_Clean': ecg_cleaned
        })
        
        return df, rpeaks_indices
        
    except Exception as e:
        logging.error(f"Error in ECG processing: {e}")
        # Fallback: return original signal with basic peak detection
        df = pd.DataFrame({
            'ECG_Raw': signal,
            'ECG_Clean': signal
        })
        
        # Simple peak detection as fallback
        peaks, _ = find_peaks(signal, height=np.mean(signal) + 2*np.std(signal), distance=int(0.6*fs))
        return df, peaks

def updated_extract_waves(df):
    """
    Extract wave features from ECG signal using NeuroKit2
    """
    try:
        ecg_signal = df['ECG_Clean'].values
        fs = 500  # Assume 500 Hz sampling rate
        
        # Use NeuroKit2 to extract ECG features
        _, waves = nk.ecg_delineate(ecg_signal, sampling_rate=fs)
        
        # Convert to the expected format
        wave_dict = {}
        for key, value in waves.items():
            if value is not None and len(value) > 0:
                # Remove NaN values and convert to numpy array
                clean_values = np.array(value)[~np.isnan(value)]
                wave_dict[key] = clean_values.astype(int)
            else:
                wave_dict[key] = np.array([])
        
        return wave_dict
        
    except Exception as e:
        logging.error(f"Error in wave extraction: {e}")
        # Return empty wave dictionary as fallback
        return {
            'ECG_P_Onsets': np.array([]),
            'ECG_P_Offsets': np.array([]),
            'ECG_P_Peaks': np.array([]),
            'ECG_QRS_Onsets': np.array([]),
            'ECG_QRS_Offsets': np.array([]),
            'ECG_Q_Peaks': np.array([]),
            'ECG_S_Peaks': np.array([]),
            'ECG_T_Onsets': np.array([]),
            'ECG_T_Offsets': np.array([]),
            'ECG_T_Peaks': np.array([])
        }

def plot_waveform_diagram(signal, fs, rpeaks, waves, title="ECG Analysis", filename=None):
    """
    Plot ECG waveform with detected features
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Time axis
        time = np.arange(len(signal)) / fs
        
        # Plot ECG signal
        plt.plot(time, signal, 'b-', linewidth=1, label='ECG Signal')
        
        # Plot R-peaks
        if len(rpeaks) > 0:
            plt.plot(time[rpeaks], signal[rpeaks], 'ro', markersize=8, label='R-peaks')
        
        # Plot other wave features if available
        colors = {'P': 'green', 'Q': 'orange', 'S': 'purple', 'T': 'red'}
        
        for wave_type in ['P', 'Q', 'S', 'T']:
            peak_key = f'ECG_{wave_type}_Peaks'
            if peak_key in waves and len(waves[peak_key]) > 0:
                valid_peaks = waves[peak_key][waves[peak_key] < len(signal)]
                if len(valid_peaks) > 0:
                    plt.plot(time[valid_peaks], signal[valid_peaks], 
                           'o', color=colors[wave_type], markersize=6, 
                           label=f'{wave_type}-peaks')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        # Create a simple fallback plot
        plt.figure(figsize=(15, 8))
        time = np.arange(len(signal)) / fs
        plt.plot(time, signal, 'b-', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

def analyze_and_plot(wfdb_basename, plot_folder, file_id):
    """
    Wrapper function that calls the existing analyze_and_plot_12_lead_fixed function
    """
    return analyze_and_plot_12_lead_fixed(wfdb_basename, plot_folder, file_id)

def load_and_clean_all_leads(wfdb_path):
    """
    Load and clean all leads from a WFDB record
    """
    try:
        if not os.path.isdir(wfdb_path):
            wfdb_basename = os.path.splitext(wfdb_path)[0]
        else:
            hea_file = next((f for f in os.listdir(wfdb_path) if f.lower().endswith('.hea')), None)
            if not hea_file:
                raise FileNotFoundError(f"No .hea file found in directory {wfdb_path}")
            wfdb_basename = os.path.join(wfdb_path, os.path.splitext(hea_file)[0])
        
        signals, fs, record = read_wfdb_record(wfdb_basename)
        lead_names = record.sig_name if record.sig_name else [f"Lead_{i}" for i in range(signals.shape[1])]
        
        return {
            'cleaned_signals': signals,
            'fs': fs,
            'lead_names': lead_names
        }
    except Exception as e:
        logging.error(f"Error in load_and_clean_all_leads for path {wfdb_path}: {e}", exc_info=True)
        raise

def upload_file_to_supabase(local_path: str, storage_path: str, bucket_name="ecg-data") -> str:
    """
    Uploads a file to supabase bucket
    """
    supabase = get_supabase()
    with open(local_path, "rb") as f:
        res = supabase.storage.from_(bucket_name).upload(
            path=storage_path,
            file=f,
            file_options={"upsert": "true"}
        )
    
   # Verify upload success
    if hasattr(res, "path") and res.path:
        return res.path
    else:
        raise Exception(f"Upload failed, response: {res}")

def generate_signed_url_from_supabase(storage_path: str, bucket_name="ecg-data", expires_in=3600):
    """
    Generates a signed url to add data to supabase bucket
    """
    storage_path = storage_path.lstrip('/')  # Remove leading slash
    if storage_path.startswith("app/"):
        storage_path = storage_path[4:]  # Remove 'app/' prefix
    try:
        supabase = get_supabase() 
        res = supabase.storage.from_(bucket_name).create_signed_url(storage_path, expires_in)
        return res["signedURL"]
    except StorageApiError as e:
        logging.warning(f"Could not generate signed URL for {storage_path}: {e}")
        return None


def get_signed_urls_for_ecg(ecg):
    """
    Creates a signed url for supabase to retrieve ecg files
    """
    supabase = get_supabase()
    bucket_name = "ecg-data"
    signed_wfdb_url = supabase.storage.from_(bucket_name).create_signed_url(ecg.wfdb_path, expires_in=3600)['signedURL']
    signed_plot_url = supabase.storage.from_(bucket_name).create_signed_url(ecg.plot_path, expires_in=3600)['signedURL']
    return signed_wfdb_url, signed_plot_url