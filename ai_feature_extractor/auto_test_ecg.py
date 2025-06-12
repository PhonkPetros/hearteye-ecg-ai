import os
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import wfdb.processing
import zipfile
import tempfile
import glob

# --- Configuration ---
MODEL_PATH = r"E:\AiModel\trained_models\ecg_segmentation_unet_best.keras"
TARGET_SAMPLING_RATE = 500  # Hz - same as training
WINDOW_LENGTH_SEC = 10  # seconds
WINDOW_LENGTH_SAMPLES = int(WINDOW_LENGTH_SEC * TARGET_SAMPLING_RATE)
NUM_LEADS = 12

# Preprocessing functions (same as training)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 1 - 1e-6
    if low >= high:
        return data

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, f0, Q, fs):
    """Applies a notch filter."""
    b_notch, a_notch = iirnotch(f0, Q, fs)
    y = filtfilt(b_notch, a_notch, data)
    return y

def preprocess_ecg_signals(signals, original_fs):
    """Preprocess ECG signals the same way as during training."""
    # Ensure we have 12 leads
    if signals.shape[1] > NUM_LEADS:
        signals = signals[:, :NUM_LEADS]
    elif signals.shape[1] < NUM_LEADS:
        padding = np.zeros((signals.shape[0], NUM_LEADS - signals.shape[1]))
        signals = np.hstack((signals, padding))

    # Apply filters
    lowcut_bp = 0.005
    highcut_bp = 150.0
    notch_freq = 60.0
    notch_Q = 30.0

    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        lead_signal = signals[:, i]
        lead_signal_bandpassed = butter_bandpass_filter(lead_signal, lowcut_bp, highcut_bp, original_fs, order=3)
        
        if original_fs > 2 * notch_freq:
            lead_signal_notched = notch_filter(lead_signal_bandpassed, notch_freq, notch_Q, original_fs)
        else:
            lead_signal_notched = lead_signal_bandpassed
        filtered_signals[:, i] = lead_signal_notched

    # Resample to target sampling rate
    if original_fs != TARGET_SAMPLING_RATE:
        num_resampled_samples = int(round(filtered_signals.shape[0] * TARGET_SAMPLING_RATE / original_fs))
        resampled_signals = np.zeros((num_resampled_samples, NUM_LEADS))
        
        for i_lead in range(NUM_LEADS):
            resampled_lead, _ = wfdb.processing.resample_sig(filtered_signals[:, i_lead], original_fs, TARGET_SAMPLING_RATE)
            
            if resampled_lead.shape[0] == num_resampled_samples:
                resampled_signals[:, i_lead] = resampled_lead
            elif resampled_lead.shape[0] > num_resampled_samples:
                resampled_signals[:, i_lead] = resampled_lead[:num_resampled_samples]
            else:
                resampled_signals[:resampled_lead.shape[0], i_lead] = resampled_lead
    else:
        resampled_signals = filtered_signals

    # Normalize
    normalized_signals = np.zeros_like(resampled_signals)
    for i in range(resampled_signals.shape[1]):
        mean = np.mean(resampled_signals[:, i])
        std = np.std(resampled_signals[:, i])
        if std > 1e-6:
            normalized_signals[:, i] = (resampled_signals[:, i] - mean) / std
        else:
            normalized_signals[:, i] = 0

    return normalized_signals

def extract_zip_file(zip_path, extract_to=None):
    """Extract a zip file and return the extraction directory."""
    if extract_to is None:
        extract_to = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to

def find_wfdb_files(directory):
    """Find WFDB files (.hea and .dat) in a directory."""
    wfdb_files = []
    for root, dirs, files in os.walk(directory):
        hea_files = [f for f in files if f.endswith('.hea')]
        for hea_file in hea_files:
            base_name = hea_file[:-4]  # Remove .hea extension
            dat_file = base_name + '.dat'
            if dat_file in files:
                wfdb_path = os.path.join(root, base_name)
                wfdb_files.append(wfdb_path)
    
    return wfdb_files

def predict_ecg_segments(model, ecg_data):
    """Predict ECG segments using the trained model."""
    # Create windows from the ECG data
    num_windows = ecg_data.shape[0] // WINDOW_LENGTH_SAMPLES
    if num_windows == 0:
        print(f"Warning: ECG data too short ({ecg_data.shape[0]} samples). Need at least {WINDOW_LENGTH_SAMPLES} samples.")
        return None, None
    
    windows = []
    for i in range(num_windows):
        start_idx = i * WINDOW_LENGTH_SAMPLES
        end_idx = start_idx + WINDOW_LENGTH_SAMPLES
        window = ecg_data[start_idx:end_idx, :]
        windows.append(window)
    
    windows = np.array(windows)
    print(f"Created {num_windows} windows of shape {windows.shape}")
    
    # Make predictions
    predictions = model.predict(windows, verbose=1)
    
    return windows, predictions

def visualize_predictions(windows, predictions, window_idx=0, lead_idx=0, save_path=None):
    """Visualize ECG signal with predicted segments."""
    if window_idx >= len(windows):
        print(f"Window index {window_idx} out of range. Available windows: 0-{len(windows)-1}")
        return
    
    window = windows[window_idx]
    prediction = predictions[window_idx]
    
    # Time axis (in seconds)
    time_axis = np.arange(WINDOW_LENGTH_SAMPLES) / TARGET_SAMPLING_RATE
    
    plt.figure(figsize=(15, 10))
    
    # Plot ECG signal
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, window[:, lead_idx], 'b-', linewidth=1)
    plt.title(f'ECG Signal - Window {window_idx}, Lead {lead_idx}')
    plt.ylabel('Amplitude (normalized)')
    plt.grid(True, alpha=0.3)
    
    # Plot P-wave prediction
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, prediction[:, 0], 'r-', linewidth=2)
    plt.title('P-wave Prediction')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Plot QRS-complex prediction
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, prediction[:, 1], 'g-', linewidth=2)
    plt.title('QRS-complex Prediction')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Plot QT-interval prediction
    plt.subplot(4, 1, 4)
    plt.plot(time_axis, prediction[:, 2], 'm-', linewidth=2)
    plt.title('QT-interval Prediction')
    plt.ylabel('Probability')
    plt.xlabel('Time (seconds)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return plt.gcf()

def test_wfdb_file(wfdb_path, model, output_prefix=""):
    """Test a WFDB file with the trained ECG segmentation model."""
    print(f"\n{'='*60}")
    print(f"Testing WFDB file: {wfdb_path}")
    print(f"{'='*60}")
    
    # Load WFDB record
    print("Loading WFDB record...")
    try:
        record = wfdb.rdrecord(wfdb_path)
        signals = record.p_signal
        original_fs = record.fs
        
        print(f"✓ Record loaded: {signals.shape[0]} samples, {signals.shape[1]} leads, {original_fs} Hz")
        print(f"✓ Duration: {signals.shape[0] / original_fs:.2f} seconds")
        
        if hasattr(record, 'sig_name'):
            print(f"✓ Lead names: {record.sig_name}")
        
    except Exception as e:
        print(f"✗ Error loading WFDB record: {e}")
        return None
    
    # Preprocess the signals
    print("\nPreprocessing signals...")
    try:
        processed_signals = preprocess_ecg_signals(signals, original_fs)
        print(f"✓ Processed signals shape: {processed_signals.shape}")
    except Exception as e:
        print(f"✗ Error preprocessing signals: {e}")
        return None
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        windows, predictions = predict_ecg_segments(model, processed_signals)
        if windows is None:
            return None
        
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ P-wave range: [{predictions[:,:,0].min():.3f}, {predictions[:,:,0].max():.3f}]")
        print(f"✓ QRS range: [{predictions[:,:,1].min():.3f}, {predictions[:,:,1].max():.3f}]")
        print(f"✓ QT range: [{predictions[:,:,2].min():.3f}, {predictions[:,:,2].max():.3f}]")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return None
    
    # Create results
    base_name = os.path.basename(wfdb_path)
    
    # Save visualization
    print("\nCreating visualization...")
    try:
        plot_path = f"{output_prefix}ecg_prediction_{base_name}.png"
        fig = visualize_predictions(windows, predictions, window_idx=0, lead_idx=0, save_path=plot_path)
        plt.close(fig)  # Close to save memory
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
    
    # Save numerical results
    print("Saving results...")
    try:
        # Save predictions as numpy arrays
        pred_path = f"{output_prefix}ecg_predictions_{base_name}.npz"
        np.savez(pred_path, 
                windows=windows, 
                predictions=predictions,
                original_signals=signals,
                processed_signals=processed_signals)
        print(f"✓ Predictions saved to: {pred_path}")
        
        # Calculate time per sample in milliseconds
        ms_per_sample = 1000.0 / TARGET_SAMPLING_RATE  # 2 ms per sample at 500 Hz
        
        # Save summary as CSV
        summary_data = []
        for i, pred in enumerate(predictions):
            # Find peaks/segments above threshold
            p_wave_prob = pred[:, 0].max()
            qrs_prob = pred[:, 1].max()
            qt_prob = pred[:, 2].max()
            
            # Find where probabilities are above threshold (0.5)
            p_detected_samples = np.where(pred[:, 0] > 0.5)[0]
            qrs_detected_samples = np.where(pred[:, 1] > 0.5)[0]
            qt_detected_samples = np.where(pred[:, 2] > 0.5)[0]
            
            # Calculate durations in milliseconds
            def calculate_segment_duration(detected_samples):
                if len(detected_samples) == 0:
                    return 0
                # Find continuous segments
                if len(detected_samples) == 1:
                    return ms_per_sample
                
                # Group consecutive samples
                segments = []
                current_segment = [detected_samples[0]]
                
                for j in range(1, len(detected_samples)):
                    if detected_samples[j] == detected_samples[j-1] + 1:
                        current_segment.append(detected_samples[j])
                    else:
                        segments.append(current_segment)
                        current_segment = [detected_samples[j]]
                segments.append(current_segment)
                
                # Return duration of longest segment
                max_segment_length = max(len(seg) for seg in segments)
                return max_segment_length * ms_per_sample
            
            p_wave_duration_ms = calculate_segment_duration(p_detected_samples)
            qrs_duration_ms = calculate_segment_duration(qrs_detected_samples)
            qt_duration_ms = calculate_segment_duration(qt_detected_samples)
            
            summary_data.append({
                'window': i,
                'p_wave_max_prob': p_wave_prob,
                'qrs_max_prob': qrs_prob,
                'qt_max_prob': qt_prob,
                'p_wave_detected': p_wave_prob > 0.5,
                'qrs_detected': qrs_prob > 0.5,
                'qt_detected': qt_prob > 0.5,
                'p_wave_duration_ms': round(p_wave_duration_ms, 1),
                'qrs_duration_ms': round(qrs_duration_ms, 1),
                'qt_duration_ms': round(qt_duration_ms, 1),
                'p_wave_samples_detected': len(p_detected_samples),
                'qrs_samples_detected': len(qrs_detected_samples),
                'qt_samples_detected': len(qt_detected_samples)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_prefix}ecg_summary_{base_name}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved to: {summary_path}")
        
        # Print summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total windows analyzed: {len(predictions)}")
        print(f"   P-waves detected: {summary_df['p_wave_detected'].sum()}/{len(predictions)} windows")
        print(f"   QRS complexes detected: {summary_df['qrs_detected'].sum()}/{len(predictions)} windows")
        print(f"   QT intervals detected: {summary_df['qt_detected'].sum()}/{len(predictions)} windows")
        print(f"   Average P-wave probability: {summary_df['p_wave_max_prob'].mean():.3f}")
        print(f"   Average QRS probability: {summary_df['qrs_max_prob'].mean():.3f}")
        print(f"   Average QT probability: {summary_df['qt_max_prob'].mean():.3f}")
        
        # Print duration statistics for detected segments
        detected_p = summary_df[summary_df['p_wave_detected']]
        detected_qrs = summary_df[summary_df['qrs_detected']]
        detected_qt = summary_df[summary_df['qt_detected']]
        
        if len(detected_p) > 0:
            print(f"   P-wave duration: {detected_p['p_wave_duration_ms'].mean():.1f} ± {detected_p['p_wave_duration_ms'].std():.1f} ms")
        if len(detected_qrs) > 0:
            print(f"   QRS duration: {detected_qrs['qrs_duration_ms'].mean():.1f} ± {detected_qrs['qrs_duration_ms'].std():.1f} ms")
        if len(detected_qt) > 0:
            print(f"   QT duration: {detected_qt['qt_duration_ms'].mean():.1f} ± {detected_qt['qt_duration_ms'].std():.1f} ms")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
    
    return {
        'wfdb_path': wfdb_path,
        'original_shape': signals.shape,
        'original_fs': original_fs,
        'num_windows': len(windows),
        'summary': summary_df
    }

def main():
    """Main function to automatically test all zip files in the directory."""
    print("🔬 === AUTOMATED ECG SEGMENTATION MODEL TESTING ===")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have trained the model first.")
        return
    
    # Load the model once
    print("🧠 Loading trained model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find all zip files in current directory
    zip_files = glob.glob("*.zip")
    
    if not zip_files:
        print("❌ No zip files found in current directory.")
        return
    
    print(f"\n📁 Found {len(zip_files)} zip files to process:")
    for i, zip_file in enumerate(zip_files):
        print(f"   {i+1}. {zip_file}")
    
    all_results = []
    
    # Process each zip file
    for i, zip_file in enumerate(zip_files):
        print(f"\n🔄 Processing zip file {i+1}/{len(zip_files)}: {zip_file}")
        
        try:
            # Extract zip file
            extract_dir = extract_zip_file(zip_file)
            print(f"📂 Extracted to: {extract_dir}")
            
            # Find WFDB files in extracted directory
            wfdb_files = find_wfdb_files(extract_dir)
            print(f"📋 Found {len(wfdb_files)} WFDB files in zip")
            
            if not wfdb_files:
                print("⚠️  No WFDB files found in this zip")
                continue
            
            # Process each WFDB file in the zip
            for j, wfdb_file in enumerate(wfdb_files):
                print(f"\n   📊 Processing WFDB file {j+1}/{len(wfdb_files)}")
                output_prefix = f"{os.path.splitext(zip_file)[0]}_"
                result = test_wfdb_file(wfdb_file, model, output_prefix)
                
                if result:
                    result['zip_file'] = zip_file
                    all_results.append(result)
                    print(f"   ✅ Successfully processed!")
                else:
                    print(f"   ❌ Failed to process this file")
            
        except Exception as e:
            print(f"❌ Error processing {zip_file}: {e}")
    
    # Create overall summary
    if all_results:
        print(f"\n🎉 === OVERALL RESULTS SUMMARY ===")
        print(f"📊 Successfully processed {len(all_results)} ECG files from {len(zip_files)} zip files")
        
        # Combine all summaries
        all_summaries = []
        for result in all_results:
            summary_with_file = result['summary'].copy()
            summary_with_file['zip_file'] = result['zip_file']
            summary_with_file['wfdb_path'] = result['wfdb_path']
            summary_with_file['original_fs'] = result['original_fs']
            summary_with_file['duration_sec'] = result['original_shape'][0] / result['original_fs']
            all_summaries.append(summary_with_file)
        
        if all_summaries:
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            combined_summary.to_csv("combined_ecg_analysis_results.csv", index=False)
            print(f"📄 Combined results saved to: combined_ecg_analysis_results.csv")
            
            # Print overall statistics
            total_windows = len(combined_summary)
            p_detected = combined_summary['p_wave_detected'].sum()
            qrs_detected = combined_summary['qrs_detected'].sum()
            qt_detected = combined_summary['qt_detected'].sum()
            
            print(f"\n📈 OVERALL STATISTICS:")
            print(f"   Total windows analyzed: {total_windows}")
            print(f"   P-waves detected: {p_detected} ({p_detected/total_windows*100:.1f}%)")
            print(f"   QRS complexes detected: {qrs_detected} ({qrs_detected/total_windows*100:.1f}%)")
            print(f"   QT intervals detected: {qt_detected} ({qt_detected/total_windows*100:.1f}%)")
            print(f"   Average P-wave confidence: {combined_summary['p_wave_max_prob'].mean():.3f}")
            print(f"   Average QRS confidence: {combined_summary['qrs_max_prob'].mean():.3f}")
            print(f"   Average QT confidence: {combined_summary['qt_max_prob'].mean():.3f}")
            
            # Print overall duration statistics
            detected_p_overall = combined_summary[combined_summary['p_wave_detected']]
            detected_qrs_overall = combined_summary[combined_summary['qrs_detected']]
            detected_qt_overall = combined_summary[combined_summary['qt_detected']]
            
            print(f"\n⏱️  DURATION MEASUREMENTS:")
            if len(detected_p_overall) > 0:
                p_mean = detected_p_overall['p_wave_duration_ms'].mean()
                p_std = detected_p_overall['p_wave_duration_ms'].std()
                p_range = f"{detected_p_overall['p_wave_duration_ms'].min():.1f}-{detected_p_overall['p_wave_duration_ms'].max():.1f}"
                print(f"   P-wave duration: {p_mean:.1f} ± {p_std:.1f} ms (range: {p_range} ms)")
            else:
                print(f"   P-wave duration: No P-waves detected")
                
            if len(detected_qrs_overall) > 0:
                qrs_mean = detected_qrs_overall['qrs_duration_ms'].mean()
                qrs_std = detected_qrs_overall['qrs_duration_ms'].std()
                qrs_range = f"{detected_qrs_overall['qrs_duration_ms'].min():.1f}-{detected_qrs_overall['qrs_duration_ms'].max():.1f}"
                print(f"   QRS duration: {qrs_mean:.1f} ± {qrs_std:.1f} ms (range: {qrs_range} ms)")
            else:
                print(f"   QRS duration: No QRS complexes detected")
                
            if len(detected_qt_overall) > 0:
                qt_mean = detected_qt_overall['qt_duration_ms'].mean()
                qt_std = detected_qt_overall['qt_duration_ms'].std()
                qt_range = f"{detected_qt_overall['qt_duration_ms'].min():.1f}-{detected_qt_overall['qt_duration_ms'].max():.1f}"
                print(f"   QT duration: {qt_mean:.1f} ± {qt_std:.1f} ms (range: {qt_range} ms)")
            else:
                print(f"   QT duration: No QT intervals detected")
    
    print(f"\n✨ Analysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 