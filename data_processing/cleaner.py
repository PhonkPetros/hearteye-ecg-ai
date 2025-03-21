import os
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.signal import butter, filtfilt

class ECGCleaner:
    """
    Class-based approach for cleaning/preprocessing ECG signals, 
    with extra print statements for debugging.
    """

    def __init__(self, fs=500, lowcut=0.5, highcut=40, order=4, baseline_cutoff=0.5):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        lowcut : float
            Lower cutoff for bandpass filter (Hz).
        highcut : float
            Upper cutoff for bandpass filter (Hz).
        order : int
            Filter order for bandpass and baseline-removal filters.
        baseline_cutoff : float
            High-pass cutoff (Hz) used to remove baseline wander.
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.baseline_cutoff = baseline_cutoff

    def remove_baseline_wander(self, ecg_signal):
        print("[DEBUG] remove_baseline_wander: Starting baseline removal.")
        nyquist = 0.5 * self.fs
        high = self.baseline_cutoff / nyquist
        b, a = butter(self.order, high, btype='high')

        before_shape = ecg_signal.shape
        print(f"   [DEBUG] Input signal shape: {before_shape}")

        if ecg_signal.ndim == 1:
            corrected = filtfilt(b, a, ecg_signal)
        else:
            corrected = np.zeros_like(ecg_signal)
            for i in range(ecg_signal.shape[1]):
                corrected[:, i] = filtfilt(b, a, ecg_signal[:, i])

        after_shape = corrected.shape
        print(f"   [DEBUG] Output (baseline-corrected) shape: {after_shape}")
        print(f"   [DEBUG] After baseline removal - min: {corrected.min():.3f}, max: {corrected.max():.3f}")
        return corrected

    def bandpass_filter_ecg(self, ecg_signal):
        print("[DEBUG] bandpass_filter_ecg: Starting bandpass filtering.")
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')

        before_shape = ecg_signal.shape
        print(f"   [DEBUG] Input signal shape: {before_shape}")

        if ecg_signal.ndim == 1:
            filtered = filtfilt(b, a, ecg_signal)
        else:
            filtered = np.zeros_like(ecg_signal)
            for i in range(ecg_signal.shape[1]):
                filtered[:, i] = filtfilt(b, a, ecg_signal[:, i])

        after_shape = filtered.shape
        print(f"   [DEBUG] Output (bandpass-filtered) shape: {after_shape}")
        print(f"   [DEBUG] After bandpass - min: {filtered.min():.3f}, max: {filtered.max():.3f}")
        return filtered

    def zscore_normalize(self, ecg_signal):
        print("[DEBUG] zscore_normalize: Starting z-score normalization.")
        before_shape = ecg_signal.shape
        print(f"   [DEBUG] Input signal shape: {before_shape}")

        if ecg_signal.ndim == 1:
            mean = np.mean(ecg_signal)
            std = np.std(ecg_signal)
            normalized = (ecg_signal - mean) / (std + 1e-8)
        else:
            normalized = np.zeros_like(ecg_signal)
            for i in range(ecg_signal.shape[1]):
                mean = np.mean(ecg_signal[:, i])
                std = np.std(ecg_signal[:, i])
                normalized[:, i] = (ecg_signal[:, i] - mean) / (std + 1e-8)

        after_shape = normalized.shape
        print(f"   [DEBUG] Output (normalized) shape: {after_shape}")
        print(f"   [DEBUG] After normalization - min: {normalized.min():.3f}, max: {normalized.max():.3f}")
        return normalized

    def clean_ecg(self, ecg_signal):
        """
        A one-stop method to do all cleaning steps in sequence:
        1) Remove baseline wander
        2) Bandpass filter
        3) Z-score normalize
        """
        print("[DEBUG] clean_ecg: Cleaning ECG signal.")
        ecg_no_baseline = self.remove_baseline_wander(ecg_signal)
        ecg_bandpassed = self.bandpass_filter_ecg(ecg_no_baseline)
        ecg_normalized = self.zscore_normalize(ecg_bandpassed)
        return ecg_normalized
