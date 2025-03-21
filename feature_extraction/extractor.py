import numpy as np
import pandas as pd
import neurokit2 as nk
import os
import warnings
import wfdb

# Suppress the pandas SettingWithCopyWarning from neurokit2
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class ECGFeatureExtractor:
    """
    Simplified class for extracting ECG features using neurokit2.
    """
    
    def __init__(self, sampling_rate=500.0):
        """
        Initialize the ECG Feature Extractor.
        
        Parameters
        ----------
        sampling_rate : float
            The sampling rate of the ECG signal in Hz (default: 500.0)
        """
        self.sampling_rate = sampling_rate
    
    def extract_from_wfdb(self, record_path):
        """
        Extract features from a WFDB record.
        
        Parameters
        ----------
        record_path : str
            Path to the WFDB record (without extension)
            
        Returns
        -------
        dict
            Dictionary containing extracted ECG features
        """
        try:
            # Load WFDB record
            signals, fields = wfdb.rdsamp(record_path)
            
            # Use lead II (index 1) for feature extraction, as it's common for ECG analysis
            # If lead II is not available, use the first lead
            if signals.shape[1] >= 2:
                ecg_signal = signals[:, 1]
            else:
                ecg_signal = signals[:, 0]
                
            return self.extract_features(ecg_signal)
        except Exception as e:
            print(f"[ERROR] Failed to extract features from WFDB: {str(e)}")
            return self._create_empty_features()
    
    def extract_features(self, ecg_signal):
        """
        Extract features from an ECG signal using a simplified approach.
        
        Parameters
        ----------
        ecg_signal : array-like
            The raw ECG signal
            
        Returns
        -------
        dict
            Dictionary containing extracted ECG features
        """
        try:
            # Process the ECG signal with more direct approach
            try:
                # Try the standard approach first
                ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate)
                rpeaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=self.sampling_rate)
                r_peaks_indices = info['ECG_R_Peaks']  # Get R-peaks indices
            except Exception as e:
                print(f"[WARNING] Standard peak detection failed: {str(e)}")
                # Fallback method with different parameters
                ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method="neurokit")
                rpeaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=self.sampling_rate, method="neurokit")
                r_peaks_indices = info['ECG_R_Peaks']  # Get R-peaks indices
                
            # Initialize features dictionary
            features = {}
            
            
            # Try to extract wave features if we have enough R-peaks
            if len(r_peaks_indices) > 3:
                try:
                    # Get wave delineation
                    waves_dict = self._extract_wave_features(ecg_clean, r_peaks_indices)
                    features.update(waves_dict)
                except Exception as e:
                    print(f"[WARNING] Wave delineation failed: {str(e)}")
                    # Add placeholder values
                    waves_dict = {
                        'QRS_Duration_Mean': np.nan,
                        'QRS_Duration_STD': np.nan,
                        'QT_Interval_Mean': np.nan,
                        'QT_Interval_STD': np.nan,
                        'PQ_Interval_Mean': np.nan,
                        'PQ_Interval_STD': np.nan,
                        'P_Duration_Mean': np.nan,
                        'P_Duration_STD': np.nan,
                    }
                    features.update(waves_dict)
            else:
                # Add placeholder values
                waves_dict = {
                    'QRS_Duration_Mean': np.nan,
                    'QRS_Duration_STD': np.nan,
                    'QT_Interval_Mean': np.nan,
                    'QT_Interval_STD': np.nan,
                    'PQ_Interval_Mean': np.nan,
                    'PQ_Interval_STD': np.nan,
                    'P_Duration_Mean': np.nan,
                    'P_Duration_STD': np.nan,
                }
                features.update(waves_dict)
                
            return features
            
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {str(e)}")
            return self._create_empty_features()
    
    def _extract_wave_features(self, ecg_clean, r_peaks):
        """
        Safely extract wave features using neurokit2.
        
        Parameters
        ----------
        ecg_clean : array-like
            Cleaned ECG signal
        r_peaks : array-like
            R-peak indices
            
        Returns
        -------
        dict
            Dictionary with wave features
        """
        try:
            # Try direct approach instead of using ecg_delineate
            # Get peaks for different waves
            waves = {}
            
            # Try to get QRS complexes
            try:
                _, waves_qrs = nk.ecg_delineate(ecg_clean, r_peaks, 
                                              sampling_rate=self.sampling_rate,
                                              method="peaks", show=False)
                waves.update(waves_qrs)
            except:
                pass
            
            # Try to get P and T waves
            try:
                _, waves_dwt = nk.ecg_delineate(ecg_clean, r_peaks, 
                                              sampling_rate=self.sampling_rate,
                                              method="dwt", show=False)
                # Only update with keys that don't exist
                for key in waves_dwt:
                    if key not in waves:
                        waves[key] = waves_dwt[key]
            except:
                pass
            
            # Initialize feature dictionary
            features = {}
            
            # Extract QRS duration
            qrs_durations = []
            if 'ECG_R_Onsets' in waves and 'ECG_R_Offsets' in waves:
                for onset, offset in zip(waves['ECG_R_Onsets'], waves['ECG_R_Offsets']):
                    if onset is not None and offset is not None:
                        qrs_durations.append((offset - onset) / self.sampling_rate * 1000)  # in ms
            
            features['QRS_Duration_Mean'] = np.nanmean(qrs_durations) if qrs_durations else np.nan
            features['QRS_Duration_STD'] = np.nanstd(qrs_durations) if qrs_durations else np.nan
            
            # Extract QT interval
            qt_intervals = []
            if 'ECG_Q_Peaks' in waves and 'ECG_T_Offsets' in waves:
                for q_peak, t_offset in zip(waves['ECG_Q_Peaks'], waves['ECG_T_Offsets']):
                    if q_peak is not None and t_offset is not None:
                        qt_intervals.append((t_offset - q_peak) / self.sampling_rate * 1000)  # in ms
            
            features['QT_Interval_Mean'] = np.nanmean(qt_intervals) if qt_intervals else np.nan
            features['QT_Interval_STD'] = np.nanstd(qt_intervals) if qt_intervals else np.nan
            
            # Extract PQ/PR interval (from P onset to QRS onset)
            pq_intervals = []
            if 'ECG_P_Onsets' in waves and 'ECG_R_Onsets' in waves:
                for p_onset, qrs_onset in zip(waves['ECG_P_Onsets'], waves['ECG_R_Onsets']):
                    if p_onset is not None and qrs_onset is not None:
                        pq_intervals.append((qrs_onset - p_onset) / self.sampling_rate * 1000)  # in ms
            
            features['PQ_Interval_Mean'] = np.nanmean(pq_intervals) if pq_intervals else np.nan
            features['PQ_Interval_STD'] = np.nanstd(pq_intervals) if pq_intervals else np.nan
            
            # Extract P wave duration
            p_durations = []
            if 'ECG_P_Onsets' in waves and 'ECG_P_Offsets' in waves:
                for onset, offset in zip(waves['ECG_P_Onsets'], waves['ECG_P_Offsets']):
                    if onset is not None and offset is not None:
                        p_durations.append((offset - onset) / self.sampling_rate * 1000)  # in ms
            
            features['P_Duration_Mean'] = np.nanmean(p_durations) if p_durations else np.nan
            features['P_Duration_STD'] = np.nanstd(p_durations) if p_durations else np.nan
            
            
            return features
            
        except Exception as e:
            print(f"[WARNING] Wave feature extraction failed: {str(e)}")
            return {
                'QRS_Duration_Mean': np.nan,
                'QRS_Duration_STD': np.nan,
                'QT_Interval_Mean': np.nan,
                'QT_Interval_STD': np.nan,
                'PQ_Interval_Mean': np.nan,
                'PQ_Interval_STD': np.nan,
                'P_Duration_Mean': np.nan,
                'P_Duration_STD': np.nan,
            }
    
    def _create_empty_features(self):
        """
        Create a dictionary with empty feature values.
        
        Returns
        -------
        dict
            Dictionary with NaN values for all features
        """
        return {
            'QRS_Duration_Mean': np.nan,
            'QRS_Duration_STD': np.nan,
            'QT_Interval_Mean': np.nan,
            'QT_Interval_STD': np.nan,
            'PQ_Interval_Mean': np.nan,
            'PQ_Interval_STD': np.nan,
            'P_Duration_Mean': np.nan,
            'P_Duration_STD': np.nan,
        }


