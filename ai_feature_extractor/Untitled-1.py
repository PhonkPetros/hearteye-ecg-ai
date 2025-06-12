import os
import pandas as pd
import numpy as np
import wfdb # For reading ECG files
from scipy.signal import butter, filtfilt, iirnotch # For basic filtering example
import wfdb.processing # For resampling
import json # For saving/loading progress

# --- Configuration ---
# Paths
LABELS_CSV_PATH = r"E:\AiModel\data_splits\train_records.csv"  # Path to your labels CSV
WFDB_BASE_PATH = r"G:\Other computers\My Laptop\data\mimic-iv-ecg-data\files"  # Base path to WFDB files
OUTPUT_DIR = r"E:\AiModel\processed_ecg_data_full"  # Output directory for processed data

# Processing parameters
TARGET_SAMPLING_RATE = 500  # Hz
WINDOW_LENGTH_SEC = 10  # seconds
WINDOW_LENGTH_SAMPLES = int(WINDOW_LENGTH_SEC * TARGET_SAMPLING_RATE)
NUM_LEADS = 12

# Limit the number of records to process (for testing/memory management)
MAX_RECORDS = 2000  # Increased from smaller number to process more records

# --- Synthetic Mask Generation Parameters ---
# P-wave parameters
P_WAVE_DURATION_RANGE = (0.08, 0.12)  # seconds
P_WAVE_AMPLITUDE_RANGE = (0.1, 0.3)   # relative to QRS

# QRS complex parameters  
QRS_DURATION_RANGE = (0.06, 0.12)     # seconds
QRS_AMPLITUDE_RANGE = (0.8, 1.0)      # relative amplitude

# QT interval parameters
QT_DURATION_RANGE = (0.35, 0.45)      # seconds
QT_AMPLITUDE_RANGE = (0.3, 0.7)       # relative to QRS

# Noise and variation
NOISE_LEVEL = 0.1
POSITION_JITTER = 0.02  # seconds

# --- Output and Progress Configuration ---
X_DATA_PATH = os.path.join(OUTPUT_DIR, "X_data.npy")
Y_DATA_PATH = os.path.join(OUTPUT_DIR, "Y_data.npy")
DF_INFO_PATH = os.path.join(OUTPUT_DIR, "df_info.csv")
PROGRESS_FILE_PATH = os.path.join(OUTPUT_DIR, "processing_progress.json")
SAVE_EVERY_N_RECORDS = 10 # How often to save progress and data

# --- Helper Functions ---

def construct_wfdb_path(base_dir, subject_id, study_id, cart_id): # cart_id is kept for now, but not used for stem
    """
    Constructs the WFDB base path (without .hea/.dat extension) for a MIMIC-IV-ECG record.
    """
    subject_id_str = str(subject_id)

    N_prefix_digits = 4
    if len(subject_id_str) >= N_prefix_digits:
        p_prefix_folder = f"p{subject_id_str[:N_prefix_digits]}"
    elif len(subject_id_str) > 0 :
        p_prefix_folder = f"p{subject_id_str.zfill(N_prefix_digits)}"
    else:
        p_prefix_folder = "p0000" # Should not happen

    p_subject_folder = f"p{subject_id_str}"
    s_study_folder = f"s{str(study_id)}"

    # --- CRITICAL PART: Determining the record_stem ---
    # Updated based on user's `ls` output: filenames are <study_id>.hea/dat
    record_stem = str(study_id)
    # --- END CRITICAL PART ---

    constructed_path = os.path.join(base_dir, p_prefix_folder, p_subject_folder, s_study_folder, record_stem)
    # print(f"DEBUG: Constructed WFDB stem: {constructed_path}") # Keep this for debugging if path issues recur

    return constructed_path


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
        # print(f"Warning: Invalid bandpass range (low: {low * nyq:.3f}Hz, high: {high * nyq:.3f}Hz for fs={fs}Hz). Returning original data.")
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
    Basic preprocessing:
    1. Select 12 leads.
    2. Apply specified bandpass and notch filters.
    3. Resample to TARGET_SAMPLING_RATE.
    4. Normalize.
    Returns processed signals and the sampling rate they are now at (current_fs).
    """
    if signals.shape[1] > NUM_LEADS:
        signals = signals[:, :NUM_LEADS]
    elif signals.shape[1] < NUM_LEADS:
        padding = np.zeros((signals.shape[0], NUM_LEADS - signals.shape[1]))
        signals = np.hstack((signals, padding))

    lowcut_bp = 0.005
    highcut_bp = 150.0
    notch_freq = 60.0
    notch_Q = 30.0

    filtered_signals_stage1 = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        lead_signal = signals[:, i]
        lead_signal_bandpassed = butter_bandpass_filter(lead_signal, lowcut_bp, highcut_bp, original_fs, order=3)

        if original_fs > 2 * notch_freq:
             lead_signal_notched = notch_filter(lead_signal_bandpassed, notch_freq, notch_Q, original_fs)
        else:
            lead_signal_notched = lead_signal_bandpassed
        filtered_signals_stage1[:, i] = lead_signal_notched

    current_fs = original_fs
    resampled_signals = filtered_signals_stage1

    if original_fs != TARGET_SAMPLING_RATE:
        num_resampled_samples = int(round(filtered_signals_stage1.shape[0] * TARGET_SAMPLING_RATE / original_fs))
        resampled_signals_temp = np.zeros((num_resampled_samples, NUM_LEADS))

        for i_lead in range(NUM_LEADS): # Changed loop variable to avoid conflict
            resampled_lead, _ = wfdb.processing.resample_sig(filtered_signals_stage1[:, i_lead], original_fs, TARGET_SAMPLING_RATE)

            if resampled_lead.shape[0] == num_resampled_samples:
                resampled_signals_temp[:, i_lead] = resampled_lead
            elif resampled_lead.shape[0] > num_resampled_samples:
                resampled_signals_temp[:, i_lead] = resampled_lead[:num_resampled_samples]
            else:
                 resampled_signals_temp[:resampled_lead.shape[0], i_lead] = resampled_lead

        resampled_signals = resampled_signals_temp
        current_fs = TARGET_SAMPLING_RATE

    normalized_signals = np.zeros_like(resampled_signals)
    for i in range(resampled_signals.shape[1]): # Changed loop variable to avoid conflict
        mean = np.mean(resampled_signals[:, i])
        std = np.std(resampled_signals[:, i])
        if std > 1e-6:
            normalized_signals[:, i] = (resampled_signals[:, i] - mean) / std
        else:
            normalized_signals[:, i] = 0

    return normalized_signals, current_fs


def create_target_masks(window_len_samples, fiducial_points_in_window):
    """
    Creates binary target masks for P-wave, QRS-complex, and QT-interval.
    Fiducial points are expected as a dictionary with keys like 'p_onset', 'p_end', etc.,
    and values are sample indices relative to the start of the current ECG window,
    at the `TARGET_SAMPLING_RATE`.
    """
    p_mask = np.zeros(window_len_samples, dtype=np.float32)
    qrs_mask = np.zeros(window_len_samples, dtype=np.float32)
    qt_mask = np.zeros(window_len_samples, dtype=np.float32)

    def get_valid_point(key):
        val = fiducial_points_in_window.get(key)
        if pd.notna(val):
            val_int = int(round(val))
            return max(0, min(val_int, window_len_samples - 1))
        return None

    p_onset = get_valid_point('p_onset')
    p_end = get_valid_point('p_end')
    qrs_onset = get_valid_point('qrs_onset')
    qrs_end = get_valid_point('qrs_end')
    t_end = get_valid_point('t_end')

    if p_onset is not None and p_end is not None and p_onset < p_end:
        p_mask[p_onset : p_end + 1] = 1.0

    if qrs_onset is not None and qrs_end is not None and qrs_onset < qrs_end:
        qrs_mask[qrs_onset : qrs_end + 1] = 1.0

    if qrs_onset is not None and t_end is not None and qrs_onset < t_end:
        qt_mask[qrs_onset : t_end + 1] = 1.0

    target_y = np.stack([p_mask, qrs_mask, qt_mask], axis=-1)
    return target_y

# --- Main Data Preparation Loop ---
def prepare_data_for_training(labels_df_subset, ecg_base_dir,
                              current_X_data_list, current_Y_data_list, current_df_info_list,
                              save_every_n, progress_file, x_out_path, y_out_path, df_info_out_path,
                              initial_global_offset_index=0):

    num_newly_processed_in_batch = 0

    # labels_df_subset.iterrows() yields (original_index, row_data)
    # The original_index is from the full labels_df if labels_df_subset was created by slicing.
    for original_index, row in labels_df_subset.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        cart_id = row['cart_id']

        try:
            wfdb_path_stem = construct_wfdb_path(ecg_base_dir, subject_id, study_id, cart_id)

            if not os.path.exists(f"{wfdb_path_stem}.hea"):
                # print(f"DEBUG: Header not found for {wfdb_path_stem}, skipping.")
                continue

            record = wfdb.rdrecord(wfdb_path_stem)
            signals_original = record.p_signal
            original_fs = record.fs

            if signals_original is None or signals_original.ndim < 2 or signals_original.shape[1] == 0:
                continue

            signals_processed_fs_target, current_fs_after_preprocessing = preprocess_ecg_signals(signals_original, original_fs)

            scale_factor_for_fiducials = 1.0
            if original_fs != current_fs_after_preprocessing:
                scale_factor_for_fiducials = current_fs_after_preprocessing / float(original_fs)

            num_samples_in_processed_signal = signals_processed_fs_target.shape[0]
            initial_X_len_for_this_record = len(current_X_data_list) # Before processing this specific record
            step_size = WINDOW_LENGTH_SAMPLES

            for i_window_start in range(0, num_samples_in_processed_signal - WINDOW_LENGTH_SAMPLES + 1, step_size):
                ecg_window = signals_processed_fs_target[i_window_start : i_window_start + WINDOW_LENGTH_SAMPLES, :]

                if ecg_window.shape[0] < WINDOW_LENGTH_SAMPLES:
                    continue

                fiducial_points_original_fs_indices = {
                    'p_onset': row['p_onset'], 'p_end': row['p_end'],
                    'qrs_onset': row['qrs_onset'], 'qrs_end': row['qrs_end'],
                    't_end': row['t_end']
                }

                fiducial_points_for_window_scaled = {}
                for key, val_orig_fs_idx in fiducial_points_original_fs_indices.items():
                    if pd.notna(val_orig_fs_idx):
                        val_scaled_fs_idx = val_orig_fs_idx * scale_factor_for_fiducials
                        val_relative_to_window = val_scaled_fs_idx - i_window_start
                        fiducial_points_for_window_scaled[key] = val_relative_to_window
                    else:
                        fiducial_points_for_window_scaled[key] = np.nan

                target_y = create_target_masks(WINDOW_LENGTH_SAMPLES, fiducial_points_for_window_scaled)

                current_X_data_list.append(ecg_window)
                current_Y_data_list.append(target_y)
                current_df_info_list.append({'subject_id': subject_id, 'study_id': study_id, 'cart_id': cart_id,
                                             'original_csv_index': original_index, # Store original index
                                             'window_start_sample': i_window_start,
                                             'wfdb_stem_used': wfdb_path_stem})

            if len(current_X_data_list) > initial_X_len_for_this_record: # If windows were added for this record
                 num_newly_processed_in_batch += 1

            if num_newly_processed_in_batch > 0 and num_newly_processed_in_batch % save_every_n == 0:
                print(f"  Processed {num_newly_processed_in_batch} new records in this batch, saving progress (total windows: {len(current_X_data_list)})...")
                try:
                    np.save(x_out_path, np.array(current_X_data_list))
                    np.save(y_out_path, np.array(current_Y_data_list))
                    pd.DataFrame(current_df_info_list).to_csv(df_info_out_path, index=False)
                    with open(progress_file, 'w') as f:
                        # Save the original_index of the record just processed. Next run will start from original_index + 1.
                        json.dump({'next_original_index_to_process': original_index + 1}, f)
                    print(f"  Progress saved. Last original CSV index processed: {original_index}.")
                except Exception as e_save_interim:
                    print(f"  ERROR saving interim progress: {e_save_interim}")

        except FileNotFoundError:
            # print(f"DEBUG: FileNotFoundError for record with original_index {original_index}, skipping.")
            pass
        except Exception as e:
            print(f"  ERROR processing record Subject {subject_id}, Study {study_id} (Original CSV Index: {original_index}): {e}")
            # import traceback
            # traceback.print_exc()

    print(f"\nFinished processing batch. Processed {num_newly_processed_in_batch} new records in this run.")
    # Final save at the end of the batch
    if num_newly_processed_in_batch > 0 : # Only save if something new was processed
        print("Performing final save for this run...")
        try:
            np.save(x_out_path, np.array(current_X_data_list))
            np.save(y_out_path, np.array(current_Y_data_list))
            pd.DataFrame(current_df_info_list).to_csv(df_info_out_path, index=False)

            if labels_df_subset.empty: 
                 last_processed_original_idx = initial_global_offset_index -1 
            elif not current_df_info_list: 
                 last_processed_original_idx = initial_global_offset_index -1 
            else: 
                 last_processed_original_idx = current_df_info_list[-1]['original_csv_index']

            with open(progress_file, 'w') as f:
                json.dump({'next_original_index_to_process': last_processed_original_idx + 1}, f)
            print(f"  Final save complete. Next run will start from original CSV index: {last_processed_original_idx + 1}.")
        except Exception as e_save_final:
            print(f"  ERROR during final save: {e_save_final}")

    return current_X_data_list, current_Y_data_list, current_df_info_list


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using WFDB_BASE_PATH: {WFDB_BASE_PATH}")
    print("Loading labels CSV...")
    try:
        labels_df_full = pd.read_csv(LABELS_CSV_PATH)
        fiducial_cols_to_check = ['p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end']
        for col in fiducial_cols_to_check:
            if col in labels_df_full.columns:
                labels_df_full[col] = pd.to_numeric(labels_df_full[col], errors='coerce')

        labels_df_full.dropna(subset=['subject_id', 'study_id', 'cart_id'] + fiducial_cols_to_check, inplace=True)
        labels_df_full['subject_id'] = labels_df_full['subject_id'].astype(np.int64)
        labels_df_full['study_id'] = labels_df_full['study_id'].astype(np.int64)
        labels_df_full['cart_id'] = labels_df_full['cart_id'].astype(np.int64)
        labels_df_full.reset_index(drop=True, inplace=True) # Ensure original index is 0-based

        print(f"Loaded and initially filtered {len(labels_df_full)} records from CSV.")
        if labels_df_full.empty:
            print("Label CSV is empty or no records with valid IDs and fiducial points. Exiting.")
            exit()
    except FileNotFoundError:
        print(f"Error: Labels CSV file not found at {LABELS_CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading or parsing CSV: {e}")
        exit()

    # --- Load progress and existing data ---
    start_original_index = 0
    processed_X_list = []
    processed_Y_list = []
    processed_info_list = []

    if os.path.exists(PROGRESS_FILE_PATH):
        try:
            with open(PROGRESS_FILE_PATH, 'r') as f:
                progress = json.load(f)
                start_original_index = progress.get('next_original_index_to_process', 0)
            print(f"Resuming from original CSV index: {start_original_index}")

            if os.path.exists(X_DATA_PATH):
                processed_X_list = list(np.load(X_DATA_PATH, allow_pickle=True)) 
            if os.path.exists(Y_DATA_PATH):
                processed_Y_list = list(np.load(Y_DATA_PATH, allow_pickle=True))
            if os.path.exists(DF_INFO_PATH):
                processed_info_list = pd.read_csv(DF_INFO_PATH).to_dict('records')

            if processed_X_list or processed_Y_list or processed_info_list:
                 print(f"Loaded {len(processed_X_list)} existing processed windows.")

        except Exception as e_load_progress:
            print(f"Could not load progress or existing data: {e_load_progress}. Starting from scratch.")
            start_original_index = 0
            processed_X_list, processed_Y_list, processed_info_list = [], [], []
    else:
        print("No progress file found. Starting from scratch.")

    # --- Determine records to process in this run ---
    if start_original_index >= len(labels_df_full):
        print("All records from CSV have already been processed according to progress file.")
        if processed_X_list: 
            final_X_data = np.array(processed_X_list)
            final_Y_data = np.array(processed_Y_list)
            final_df_info = pd.DataFrame(processed_info_list)
            if final_X_data.size > 0: # Check if array is not empty
                print(f"\nFinal loaded data shapes: X: {final_X_data.shape}, Y: {final_Y_data.shape}, Info: {final_df_info.shape}")
        exit()

    labels_df_to_process_this_run = labels_df_full.iloc[start_original_index:]

    max_new_records_this_run = MAX_RECORDS

    if max_new_records_this_run is not None and len(labels_df_to_process_this_run) > max_new_records_this_run:
        labels_df_to_process_this_run = labels_df_to_process_this_run.head(max_new_records_this_run)

    if labels_df_to_process_this_run.empty:
        print("No new records to process in this run based on start_index and max_new_records_this_run.")
        if processed_X_list: 
            final_X_data = np.array(processed_X_list)
            final_Y_data = np.array(processed_Y_list)
            final_df_info = pd.DataFrame(processed_info_list)
            if final_X_data.size > 0:
                 print(f"\nStats for previously processed data: X: {final_X_data.shape}, Y: {final_Y_data.shape}, Info: {final_df_info.shape}")
        exit()

    print(f"\nStarting data preparation pipeline for {len(labels_df_to_process_this_run)} records (or up to limit).")

    final_X_list, final_Y_list, final_info_list = prepare_data_for_training(
        labels_df_to_process_this_run,
        WFDB_BASE_PATH,
        processed_X_list, 
        processed_Y_list,
        processed_info_list,
        SAVE_EVERY_N_RECORDS,
        PROGRESS_FILE_PATH,
        X_DATA_PATH, Y_DATA_PATH, DF_INFO_PATH,
        initial_global_offset_index=start_original_index 
    )

    final_X_data = np.array(final_X_list) if final_X_list else np.array([]) 
    final_Y_data = np.array(final_Y_list) if final_Y_list else np.array([])
    final_df_info = pd.DataFrame(final_info_list) if final_info_list else pd.DataFrame()


    if final_X_data.size > 0 and final_Y_data.size > 0:
        print(f"\nData preparation pipeline finished.")
        if not final_df_info.empty:
            print(f"Total unique ECG records processed across all runs (estimated from df_info): {final_df_info.drop_duplicates(subset=['subject_id', 'study_id', 'cart_id']).shape[0]}")
        else:
            print("No unique ECG records information available in df_info.")
        print(f"Total ECG windows generated (X_data): {final_X_data.shape}")
        print(f"Total Target masks generated (Y_data): {final_Y_data.shape}")
        print(f"Info DataFrame for processed windows (df_info): {final_df_info.shape}")

        if not final_df_info.empty and final_X_data.size > 0: # Ensure X_data is not empty for indexing
            print("\nExample of first processed record info (overall):")
            print(final_df_info.head(1))
            print("\nExample of first ECG window (overall, first lead, first 10 samples):")
            print(final_X_data[0, :10, 0])
            print("\nExample of corresponding target mask (overall, first 10 samples, P, QRS, QT):")
            print(final_Y_data[0, :10, :])
    else:
        print("\nNo data was ultimately processed or loaded. Check paths, file existence, and logs for errors.")

    num_records_in_csv_slice = len(labels_df_to_process_this_run)
    print(f"\nAttempted to process {num_records_in_csv_slice} records in this specific run (from the current slice of the CSV).")
    if not final_df_info.empty:
        num_unique_successfully_processed_total = final_df_info.drop_duplicates(subset=['subject_id', 'study_id', 'cart_id']).shape[0]
        print(f"{num_unique_successfully_processed_total} unique records have resulted in processed data across all runs.")