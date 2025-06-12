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
    """
    Preprocess ECG signals the same way as during training.
    """
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
    """
    Predict ECG segments using the trained model.
    Returns predictions for P-wave, QRS-complex, and QT-interval.
    """
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

def visualize_predictions(windows, predictions, window_idx=0, lead_idx=0):
    """
    Visualize ECG signal with predicted segments.
    """
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
    return plt.gcf()

def test_wfdb_file(wfdb_path, model_path=MODEL_PATH, visualize=True, save_results=True):
    """
    Test a WFDB file with the trained ECG segmentation model.
    
    Args:
        wfdb_path: Path to WFDB file (without extension)
        model_path: Path to trained model
        visualize: Whether to create visualizations
        save_results: Whether to save results to files
    
    Returns:
        Dictionary with results
    """
    print(f"Testing WFDB file: {wfdb_path}")
    
    # Load the trained model
    print("Loading trained model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load WFDB record
    print("Loading WFDB record...")
    try:
        record = wfdb.rdrecord(wfdb_path)
        signals = record.p_signal
        original_fs = record.fs
        
        print(f"Record loaded: {signals.shape[0]} samples, {signals.shape[1]} leads, {original_fs} Hz")
        print(f"Duration: {signals.shape[0] / original_fs:.2f} seconds")
        
        if hasattr(record, 'sig_name'):
            print(f"Lead names: {record.sig_name}")
        
    except Exception as e:
        print(f"Error loading WFDB record: {e}")
        return None
    
    # Preprocess the signals
    print("Preprocessing signals...")
    try:
        processed_signals = preprocess_ecg_signals(signals, original_fs)
        print(f"Processed signals shape: {processed_signals.shape}")
    except Exception as e:
        print(f"Error preprocessing signals: {e}")
        return None
    
    # Make predictions
    print("Making predictions...")
    try:
        windows, predictions = predict_ecg_segments(model, processed_signals)
        if windows is None:
            return None
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction ranges - P-wave: [{predictions[:,:,0].min():.3f}, {predictions[:,:,0].max():.3f}]")
        print(f"Prediction ranges - QRS: [{predictions[:,:,1].min():.3f}, {predictions[:,:,1].max():.3f}]")
        print(f"Prediction ranges - QT: [{predictions[:,:,2].min():.3f}, {predictions[:,:,2].max():.3f}]")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None
    
    # Create results dictionary
    results = {
        'wfdb_path': wfdb_path,
        'original_shape': signals.shape,
        'original_fs': original_fs,
        'processed_shape': processed_signals.shape,
        'num_windows': len(windows),
        'windows': windows,
        'predictions': predictions,
        'lead_names': getattr(record, 'sig_name', None)
    }
    
    # Visualize results
    if visualize:
        print("Creating visualizations...")
        try:
            # Create visualization for first window, first lead
            fig = visualize_predictions(windows, predictions, window_idx=0, lead_idx=0)
            
            if save_results:
                base_name = os.path.basename(wfdb_path)
                plot_path = f"ecg_prediction_{base_name}_window0_lead0.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {plot_path}")
                results['plot_path'] = plot_path
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    # Save numerical results
    if save_results:
        try:
            base_name = os.path.basename(wfdb_path)
            
            # Save predictions as numpy arrays
            pred_path = f"ecg_predictions_{base_name}.npz"
            np.savez(pred_path, 
                    windows=windows, 
                    predictions=predictions,
                    original_signals=signals,
                    processed_signals=processed_signals)
            print(f"Predictions saved to: {pred_path}")
            results['predictions_path'] = pred_path
            
            # Save summary as CSV
            summary_data = []
            for i, pred in enumerate(predictions):
                # Find peaks/segments above threshold
                p_wave_prob = pred[:, 0].max()
                qrs_prob = pred[:, 1].max()
                qt_prob = pred[:, 2].max()
                
                summary_data.append({
                    'window': i,
                    'p_wave_max_prob': p_wave_prob,
                    'qrs_max_prob': qrs_prob,
                    'qt_max_prob': qt_prob,
                    'p_wave_detected': p_wave_prob > 0.5,
                    'qrs_detected': qrs_prob > 0.5,
                    'qt_detected': qt_prob > 0.5
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = f"ecg_summary_{base_name}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to: {summary_path}")
            results['summary_path'] = summary_path
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return results

def main():
    """
    Main function to test ECG model.
    You can modify this to test your specific WFDB file.
    """
    print("=== ECG Segmentation Model Testing ===")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have trained the model first.")
        return
    
    # Example usage - you'll need to provide the path to your WFDB file
    print("\nTo test your WFDB file, you have several options:")
    print("1. If you have a zip file, extract it first")
    print("2. Provide the path to your WFDB file (without .hea/.dat extension)")
    print("3. Use the test_wfdb_file() function")
    
    # Example for zip file
    zip_file_path = input("\nEnter path to your WFDB zip file (or press Enter to skip): ").strip()
    
    if zip_file_path and os.path.exists(zip_file_path):
        print(f"Extracting zip file: {zip_file_path}")
        extract_dir = extract_zip_file(zip_file_path)
        print(f"Extracted to: {extract_dir}")
        
        # Find WFDB files in extracted directory
        wfdb_files = find_wfdb_files(extract_dir)
        print(f"Found {len(wfdb_files)} WFDB files:")
        
        for i, wfdb_file in enumerate(wfdb_files):
            print(f"  {i}: {wfdb_file}")
        
        if wfdb_files:
            # Test the first WFDB file found
            results = test_wfdb_file(wfdb_files[0])
            if results:
                print("\n=== Test Results ===")
                print(f"Successfully processed: {results['wfdb_path']}")
                print(f"Original signal: {results['original_shape']} at {results['original_fs']} Hz")
                print(f"Processed into {results['num_windows']} windows")
                print("Check the generated files for detailed results!")
    else:
        # Direct WFDB file path
        wfdb_path = input("Enter path to your WFDB file (without extension): ").strip()
        if wfdb_path and os.path.exists(wfdb_path + '.hea'):
            results = test_wfdb_file(wfdb_path)
            if results:
                print("\n=== Test Results ===")
                print(f"Successfully processed: {results['wfdb_path']}")
                print(f"Original signal: {results['original_shape']} at {results['original_fs']} Hz")
                print(f"Processed into {results['num_windows']} windows")
        else:
            print("No valid WFDB file provided or file not found.")

if __name__ == "__main__":
    main() 