import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk
from scipy.signal import butter, filtfilt

class ECGDebugger:
    """
    Debugging utility for ECG processing.
    """
    def __init__(self, record_path=None):
        """
        Initialize with an optional record path.
        """
        self.record_path = record_path
        self.signals = None
        self.fs = None
        self.fields = None
        self.results_df = None
    
    def load_record(self, record_path=None):
        """
        Load a WFDB record, trying to handle errors gracefully.
        """
        if record_path:
            self.record_path = record_path
        
        if not self.record_path:
            print("ERROR: No record path provided")
            return False
        
        try:
            print(f"Loading record: {self.record_path}")
            self.signals, self.fields = wfdb.rdsamp(self.record_path)
            self.fs = self.fields.get('fs', 500)
            
            print(f"Record loaded successfully:")
            print(f"  Sampling rate: {self.fs} Hz")
            print(f"  Signal shape: {self.signals.shape}")
            print(f"  Available channels: {self.fields['sig_name']}")
            return True
        except Exception as e:
            print(f"ERROR loading record: {str(e)}")
            return False
    
    def load_results_csv(self, csv_path):
        """
        Load previously generated results for review.
        """
        try:
            self.results_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.results_df)} records from {csv_path}")
            return True
        except Exception as e:
            print(f"ERROR loading CSV: {str(e)}")
            return False
    
    def analyze_results(self):
        """
        Analyze feature extraction results to identify potential issues.
        """
        if self.results_df is None:
            print("No results loaded. Use load_results_csv() first.")
            return
        
        print("\n=== FEATURE EXTRACTION RESULTS ANALYSIS ===")
        
        # Get statistical overview
        for col in ['QRS_duration', 'QT_duration', 'PQ_interval', 'P_duration']:
            if col in self.results_df.columns:
                data = self.results_df[col].dropna()
                if len(data) == 0:
                    print(f"{col}: No valid data")
                    continue
                    
                print(f"\n{col}:")
                print(f"  Count: {len(data)}")
                print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
                print(f"  Mean: {data.mean():.2f}, Median: {data.median():.2f}")
                print(f"  Std Dev: {data.std():.2f}")
                
                # Check for physiologically implausible values
                if col == 'QRS_duration':
                    abnormal = data[data > 200].shape[0]
                    extreme = data[data > 500].shape[0]
                    negative = data[data < 0].shape[0]
                    print(f"  Abnormal (>200ms): {abnormal} ({abnormal/len(data)*100:.1f}%)")
                    print(f"  Extreme (>500ms): {extreme} ({extreme/len(data)*100:.1f}%)")
                    print(f"  Negative: {negative} ({negative/len(data)*100:.1f}%)")
                
                elif col == 'QT_duration':
                    abnormal = data[data > 600].shape[0]
                    extreme = data[data > 1000].shape[0]
                    negative = data[data < 0].shape[0]
                    print(f"  Abnormal (>600ms): {abnormal} ({abnormal/len(data)*100:.1f}%)")
                    print(f"  Extreme (>1000ms): {extreme} ({extreme/len(data)*100:.1f}%)")
                    print(f"  Negative: {negative} ({negative/len(data)*100:.1f}%)")
                
                elif col == 'PQ_interval':
                    abnormal = data[data > 250].shape[0]
                    negative = data[data < 0].shape[0]
                    print(f"  Abnormal (>250ms): {abnormal} ({abnormal/len(data)*100:.1f}%)")
                    print(f"  Negative: {negative} ({negative/len(data)*100:.1f}%)")
                
                elif col == 'P_duration':
                    abnormal = data[data > 150].shape[0]
                    negative = data[data < 0].shape[0]
                    print(f"  Abnormal (>150ms): {abnormal} ({abnormal/len(data)*100:.1f}%)")
                    print(f"  Negative: {negative} ({negative/len(data)*100:.1f}%)")
                
        # Print records with extreme values
        self._highlight_extreme_records()
    
    def _highlight_extreme_records(self):
        """
        Find and print records with the most extreme values.
        """
        print("\n=== RECORDS WITH EXTREME VALUES ===")
        
        if 'QRS_duration' in self.results_df.columns:
            extreme_qrs = self.results_df.loc[self.results_df['QRS_duration'] > 500]
            if not extreme_qrs.empty:
                print(f"\nTop 5 extreme QRS duration values:")
                print(extreme_qrs.sort_values('QRS_duration', ascending=False)[
                    ['record', 'QRS_duration']].head(5).to_string(index=False))
        
        if 'QT_duration' in self.results_df.columns:
            extreme_qt = self.results_df.loc[self.results_df['QT_duration'] > 1000]
            if not extreme_qt.empty:
                print(f"\nTop 5 extreme QT duration values:")
                print(extreme_qt.sort_values('QT_duration', ascending=False)[
                    ['record', 'QT_duration']].head(5).to_string(index=False))
        
        if 'PQ_interval' in self.results_df.columns:
            negative_pq = self.results_df.loc[self.results_df['PQ_interval'] < 0]
            if not negative_pq.empty:
                print(f"\nSample of negative PQ intervals:")
                print(negative_pq.sort_values('PQ_interval')[
                    ['record', 'PQ_interval']].head(5).to_string(index=False))
    
    def test_ecg_processing(self, channel_idx=0, clean_method='neurokit', delineate_method='dwt'):
        """
        Test the ECG processing pipeline on a single channel.
        """
        if self.signals is None:
            print("No ECG data loaded. Use load_record() first.")
            return
        
        if channel_idx >= self.signals.shape[1]:
            print(f"Invalid channel index: {channel_idx}, max is {self.signals.shape[1]-1}")
            return
        
        print(f"\n=== TESTING PROCESSING ON CHANNEL {channel_idx} ({self.fields['sig_name'][channel_idx]}) ===")
        
        # Extract single channel
        raw_signal = self.signals[:, channel_idx]
        
        # Plot raw signal
        plt.figure(figsize=(12, 4))
        plt.plot(raw_signal)
        plt.title(f"Raw ECG Signal - Channel {self.fields['sig_name'][channel_idx]}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Clean signal
        print("\nCleaning signal...")
        try:
            cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=self.fs, method=clean_method)
            print("Cleaning successful")
            
            # Plot cleaned signal
            plt.figure(figsize=(12, 4))
            plt.plot(cleaned_signal)
            plt.title(f"Cleaned ECG Signal - Channel {self.fields['sig_name'][channel_idx]}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"ERROR during cleaning: {str(e)}")
            cleaned_signal = raw_signal
        
        # Detect R-peaks
        print("\nDetecting R-peaks...")
        try:
            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=self.fs)
            rpeaks_indices = rpeaks.get("ECG_R_Peaks", [])
            print(f"Found {len(rpeaks_indices)} R-peaks")
            
            # Plot signal with R-peaks
            plt.figure(figsize=(12, 4))
            plt.plot(cleaned_signal)
            plt.scatter(rpeaks_indices, cleaned_signal[rpeaks_indices], color='red')
            plt.title(f"ECG with R-peaks - Channel {self.fields['sig_name'][channel_idx]}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"ERROR during R-peak detection: {str(e)}")
            rpeaks_indices = []
        
        if len(rpeaks_indices) == 0:
            print("No R-peaks found, cannot continue with delineation")
            return
        
        # Wave delineation
        print("\nPerforming wave delineation...")
        try:
            _, waves = nk.ecg_delineate(
                cleaned_signal,
                rpeaks=rpeaks,
                sampling_rate=self.fs,
                method=delineate_method,
                check=True
            )
            print("Wave delineation keys:", list(waves.keys()))
            
            # Plot delineated waves if available
            self._plot_delineated_waves(cleaned_signal, waves, rpeaks_indices)
            
            # Calculate intervals
            measured_intervals = self._calculate_intervals(waves, rpeaks_indices)
            
            # Print results
            print("\nMeasured intervals (average across beats):")
            for feature, value in measured_intervals.items():
                if value is not None:
                    print(f"  {feature}: {value:.2f} ms")
                else:
                    print(f"  {feature}: None")
            
        except Exception as e:
            print(f"ERROR during wave delineation: {str(e)}")
    
    def _plot_delineated_waves(self, signal, waves, rpeaks):
        """
        Plot the signal with delineated waves.
        """
        # Choose a segment around a middle R-peak to visualize
        if len(rpeaks) < 3:
            center_idx = 0
        else:
            center_idx = rpeaks[len(rpeaks) // 2]
        
        window = int(self.fs)  # 1 second window
        start_idx = max(0, center_idx - window)
        end_idx = min(len(signal), center_idx + window)
        
        plt.figure(figsize=(12, 6))
        plt.plot(signal[start_idx:end_idx])
        
        # Plot R-peaks in window
        r_in_window = [r for r in rpeaks if start_idx <= r < end_idx]
        plt.scatter([r-start_idx for r in r_in_window],
                   [signal[r] for r in r_in_window],
                   color='red', label='R-peaks')
        
        # Plot wave onsets and offsets
        wave_types = {
            'P': ['ECG_P_Peaks', 'ECG_P_Onsets', 'ECG_P_Offsets'],
            'Q': ['ECG_Q_Peaks'],
            'S': ['ECG_S_Peaks'],
            'T': ['ECG_T_Peaks', 'ECG_T_Onsets', 'ECG_T_Offsets']
        }
        
        colors = {'P': 'green', 'Q': 'orange', 'S': 'purple', 'T': 'brown'}
        markers = {'Peaks': 'o', 'Onsets': '<', 'Offsets': '>'}
        
        for wave, wave_keys in wave_types.items():
            for key in wave_keys:
                if key in waves and len(waves[key]) > 0:
                    # Filter points to those in our window
                    points = [p for p in waves[key] if start_idx <= p < end_idx]
                    if points:
                        marker_type = next((m for m in markers if m in key), 'x')
                        plt.scatter([p-start_idx for p in points],
                                   [signal[p] for p in points],
                                   color=colors[wave], marker=markers.get(marker_type, 'x'),
                                   label=key)
        
        plt.title('ECG with Delineated Waves (Zoomed to 2 seconds)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def _calculate_intervals(self, waves, rpeaks_indices):
        """
        Calculate ECG intervals based on delineated waves.
        """
        durations = {
            "QRS_duration": None,
            "QT_duration": None,
            "PQ_interval": None,
            "P_duration": None,
        }
        
        # Skip if no R-peaks
        if not rpeaks_indices:
            return durations
        
        qrs_durations = []
        qt_intervals = []
        pq_intervals = []
        p_durations = []
        
        # The following debug messages will help identify potential issues
        print("\nDelineation Feature Counts:")
        for key in waves.keys():
            if isinstance(waves[key], list) or hasattr(waves[key], '__len__'):
                print(f"  {key}: {len(waves[key])}")
        
        # Check if key features are present in reasonable numbers
        key_check = {
            'ECG_R_Onsets': len(waves.get('ECG_R_Onsets', [])),
            'ECG_R_Offsets': len(waves.get('ECG_R_Offsets', [])),
            'ECG_P_Onsets': len(waves.get('ECG_P_Onsets', [])),
            'ECG_P_Offsets': len(waves.get('ECG_P_Offsets', [])),
            'ECG_T_Offsets': len(waves.get('ECG_T_Offsets', []))
        }
        
        print("\nFeature Counts Relative to R-peaks:")
        for key, count in key_check.items():
            expected = len(rpeaks_indices)
            print(f"  {key}: {count}/{expected} ({count/expected*100:.1f}%)")
        
        for idx, r_idx in enumerate(rpeaks_indices):
            # Onset/offset for QRS
            R_onset_idx = None
            R_offset_idx = None

            if "ECG_R_Onsets" in waves and waves["ECG_R_Onsets"]:
                R_onsets = waves["ECG_R_Onsets"]
                R_onset_candidates = [on for on in R_onsets if on <= r_idx]
                if R_onset_candidates:
                    R_onset_idx = max(R_onset_candidates)
            
            if "ECG_R_Offsets" in waves and waves["ECG_R_Offsets"]:
                R_offsets = waves["ECG_R_Offsets"]
                R_offset_candidates = [off for off in R_offsets if off >= r_idx]
                if R_offset_candidates:
                    R_offset_idx = min(R_offset_candidates)

            # Fallback to Q or S if needed for QRS calculation
            Q_idx = None
            S_idx = None
            
            if R_onset_idx is None and "ECG_Q_Peaks" in waves and waves["ECG_Q_Peaks"]:
                Q_candidates = [q for q in waves["ECG_Q_Peaks"] if q < r_idx]
                if Q_candidates:
                    Q_idx = max(Q_candidates)
                    R_onset_idx = Q_idx
            
            if R_offset_idx is None and "ECG_S_Peaks" in waves and waves["ECG_S_Peaks"]:
                S_candidates = [s for s in waves["ECG_S_Peaks"] if s > r_idx]
                if S_candidates:
                    S_idx = min(S_candidates)
                    R_offset_idx = S_idx

            # P wave info
            P_onset_idx = None
            P_offset_idx = None
            
            if "ECG_P_Onsets" in waves and waves["ECG_P_Onsets"]:
                P_candidates = [p for p in waves["ECG_P_Onsets"] if p < r_idx]
                if P_candidates:
                    P_onset_idx = max(P_candidates)
            
            if "ECG_P_Offsets" in waves and waves["ECG_P_Offsets"]:
                P_candidates = [p for p in waves["ECG_P_Offsets"] if p < r_idx]
                if P_candidates:
                    P_offset_idx = max(P_candidates)

            # T wave info
            T_offset_idx = None
            
            if "ECG_T_Offsets" in waves and waves["ECG_T_Offsets"]:
                T_candidates = [t for t in waves["ECG_T_Offsets"] if t > r_idx]
                if T_candidates:
                    T_offset_idx = min(T_candidates)

            # Calculate durations (ms)
            if R_onset_idx is not None and R_offset_idx is not None:
                dist = R_offset_idx - R_onset_idx
                # Check for physiologically plausible values (typically < 200ms)
                if 0 < dist < self.fs * 0.2:  # Less than 200ms
                    qrs_ms = dist * (1000.0 / self.fs)
                    qrs_durations.append(qrs_ms)
                else:
                    print(f"  Rejected implausible QRS duration: {dist * (1000.0 / self.fs):.2f}ms")
            
            if R_onset_idx is not None and T_offset_idx is not None:
                dist = T_offset_idx - R_onset_idx
                # Check for physiologically plausible values (typically < 600ms)
                if 0 < dist < self.fs * 0.6:  # Less than 600ms
                    qt_ms = dist * (1000.0 / self.fs)
                    qt_intervals.append(qt_ms)
                else:
                    print(f"  Rejected implausible QT interval: {dist * (1000.0 / self.fs):.2f}ms")
            
            if P_onset_idx is not None and R_onset_idx is not None:
                dist = R_onset_idx - P_onset_idx
                # Check for physiologically plausible values (typically < 250ms)
                if 0 < dist < self.fs * 0.25:  # Less than 250ms
                    pq_ms = dist * (1000.0 / self.fs)
                    pq_intervals.append(pq_ms)
                else:
                    print(f"  Rejected implausible PQ interval: {dist * (1000.0 / self.fs):.2f}ms")
            
            if P_onset_idx is not None and P_offset_idx is not None:
                dist = P_offset_idx - P_onset_idx
                # Check for physiologically plausible values (typically < 150ms)
                if 0 < dist < self.fs * 0.15:  # Less than 150ms
                    p_ms = dist * (1000.0 / self.fs)
                    p_durations.append(p_ms)
                else:
                    print(f"  Rejected implausible P duration: {dist * (1000.0 / self.fs):.2f}ms")

        # Average across beats
        if qrs_durations:
            durations["QRS_duration"] = float(np.mean(qrs_durations))
        if qt_intervals:
            durations["QT_duration"] = float(np.mean(qt_intervals))
        if pq_intervals:
            durations["PQ_interval"] = float(np.mean(pq_intervals))
        if p_durations:
            durations["P_duration"] = float(np.mean(p_durations))

        return durations

    def test_lead_selection(self):
        """
        Test different leads to find the best one for feature extraction.
        """
        if self.signals is None:
            print("No ECG data loaded. Use load_record() first.")
            return
        
        print("\n=== TESTING LEAD SELECTION ===")
        print("Processing each lead to determine best for feature extraction...\n")
        
        results = []
        
        for i in range(self.signals.shape[1]):
            lead_name = self.fields['sig_name'][i]
            print(f"Processing lead {i}: {lead_name}")
            
            # Extract signal
            raw_signal = self.signals[:, i]
            
            # Clean signal
            try:
                cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=self.fs)
            except Exception as e:
                print(f"  ERROR during cleaning: {str(e)}")
                continue
            
            # Detect R-peaks
            try:
                _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=self.fs)
                rpeaks_indices = rpeaks.get("ECG_R_Peaks", [])
                print(f"  Found {len(rpeaks_indices)} R-peaks")
            except Exception as e:
                print(f"  ERROR during R-peak detection: {str(e)}")
                continue
            
            if len(rpeaks_indices) == 0:
                print("  No R-peaks found, skipping lead")
                continue
            
            # Simple metric: signal-to-noise ratio around R-peaks
            try:
                # Calculate a simple SNR
                window = int(0.05 * self.fs)  # 50ms window
                peak_amplitudes = []
                noise_estimates = []
                
                for r_idx in rpeaks_indices:
                    if r_idx < window or r_idx >= len(cleaned_signal) - window:
                        continue
                    
                    # Peak value
                    peak_value = cleaned_signal[r_idx]
                    
                    # Noise estimate: standard deviation in windows before and after peak
                    pre_window = cleaned_signal[r_idx - window:r_idx - int(window/4)]
                    post_window = cleaned_signal[r_idx + int(window/4):r_idx + window]
                    noise_value = np.std(np.concatenate((pre_window, post_window)))
                    
                    peak_amplitudes.append(abs(peak_value))
                    noise_estimates.append(noise_value)
                
                if peak_amplitudes and noise_estimates:
                    mean_peak = np.mean(peak_amplitudes)
                    mean_noise = np.mean(noise_estimates)
                    snr = mean_peak / (mean_noise + 1e-10)
                    print(f"  SNR: {snr:.2f}")
                else:
                    snr = 0
                    print("  Could not calculate SNR")
                
                # Try wave delineation
                try:
                    _, waves = nk.ecg_delineate(
                        cleaned_signal,
                        rpeaks=rpeaks,
                        sampling_rate=self.fs,
                        method='dwt',
                        check=True
                    )
                    
                    # Count wave features
                    wave_counts = {
                        'R_peaks': len(rpeaks_indices),
                        'P_peaks': len(waves.get('ECG_P_Peaks', [])),
                        'T_peaks': len(waves.get('ECG_T_Peaks', [])),
                        'P_onsets': len(waves.get('ECG_P_Onsets', [])),
                        'P_offsets': len(waves.get('ECG_P_Offsets', [])),
                        'R_onsets': len(waves.get('ECG_R_Onsets', [])),
                        'R_offsets': len(waves.get('ECG_R_Offsets', [])),
                        'T_offsets': len(waves.get('ECG_T_Offsets', []))
                    }
                    
                    # Feature completeness score
                    completeness = sum([
                        wave_counts['P_onsets'] / max(1, wave_counts['R_peaks']),
                        wave_counts['P_offsets'] / max(1, wave_counts['R_peaks']),
                        wave_counts['R_onsets'] / max(1, wave_counts['R_peaks']),
                        wave_counts['R_offsets'] / max(1, wave_counts['R_peaks']),
                        wave_counts['T_offsets'] / max(1, wave_counts['R_peaks'])
                    ]) / 5.0
                    
                    print(f"  Feature completeness: {completeness:.2f}")
                    
                    # Calculate intervals
                    durations = self._calculate_intervals(waves, rpeaks_indices)
                    
                    # Count valid intervals
                    valid_intervals = sum(1 for v in durations.values() if v is not None)
                    print(f"  Valid intervals: {valid_intervals}/4")
                    
                    results.append({
                        'lead_idx': i,
                        'lead_name': lead_name,
                        'snr': snr,
                        'r_peaks': len(rpeaks_indices),
                        'completeness': completeness,
                        'valid_intervals': valid_intervals,
                        'durations': durations
                    })
                    
                except Exception as e:
                    print(f"  ERROR during wave delineation: {str(e)}")
            
            except Exception as e:
                print(f"  ERROR calculating SNR: {str(e)}")
        
        # Rank leads
        if results:
            print("\n=== LEAD RANKING ===")
            # Sort by various metrics
            snr_rank = sorted(results, key=lambda x: x['snr'], reverse=True)
            completeness_rank = sorted(results, key=lambda x: x['completeness'], reverse=True)
            interval_rank = sorted(results, key=lambda x: x['valid_intervals'], reverse=True)
            
            # Print top 3 for each metric
            print("\nTop leads by SNR:")
            for i, result in enumerate(snr_rank[:3]):
                print(f"  {i+1}. Lead {result['lead_name']} (SNR: {result['snr']:.2f})")
            
            print("\nTop leads by feature completeness:")
            for i, result in enumerate(completeness_rank[:3]):
                print(f"  {i+1}. Lead {result['lead_name']} (Completeness: {result['completeness']:.2f})")
            
            print("\nTop leads by valid intervals:")
            for i, result in enumerate(interval_rank[:3]):
                print(f"  {i+1}. Lead {result['lead_name']} (Valid intervals: {result['valid_intervals']}/4)")
            
            # Provide overall recommendation
            print("\nRECOMMENDED LEAD FOR FEATURE EXTRACTION:")
            # Simple scoring: 3 points for #1 rank, 2 for #2, 1 for #3
            lead_scores = {}
            for i, result in enumerate(snr_rank[:3]):
                lead_id = result['lead_idx']
                lead_scores[lead_id] = lead_scores.get(lead_id, 0) + (3 - i)
            
            for i, result in enumerate(completeness_rank[:3]):
                lead_id = result['lead_idx']
                lead_scores[lead_id] = lead_scores.get(lead_id, 0) + (3 - i)
            
            for i, result in enumerate(interval_rank[:3]):
                lead_id = result['lead_idx']
                lead_scores[lead_id] = lead_scores.get(lead_id, 0) + (3 - i)
            
            # Get best lead
            if lead_scores:
                best_lead_id = max(lead_scores.items(), key=lambda x: x[1])[0]
                best_lead = next(r for r in results if r['lead_idx'] == best_lead_id)
                
                print(f"  Lead {best_lead['lead_name']} (index {best_lead_id})")
                print(f"  SNR: {best_lead['snr']:.2f}")
                print(f"  Feature completeness: {best_lead['completeness']:.2f}")
                print(f"  Valid intervals: {best_lead['valid_intervals']}/4")
                
                # Print interval values
                print("\nIntervals from best lead:")
                for feature, value in best_lead['durations'].items():
                    if value is not None:
                        print(f"  {feature}: {value:.2f} ms")
                    else:
                        print(f"  {feature}: None")
            else:
                print("  Could not determine best lead")
        else:
            print("No valid results obtained from any lead")