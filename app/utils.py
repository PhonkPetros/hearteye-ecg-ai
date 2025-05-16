import os
import logging
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Physiologic plausible ranges (ms)
PLAUSIBLE_RANGES = {
    'P_wave_duration_ms': (60, 120),
    'PQ_interval_ms':     (50, 200),
    'QRS_duration_ms':    (60, 120),
    'QT_interval_ms':     (300, 450),
}

def read_wfdb_record(wfdb_basename):
    record = wfdb.rdrecord(wfdb_basename, physical=True)
    signals = record.p_signal if record.p_signal is not None else record.d_signal
    if signals is None:
        raise RuntimeError("No signal data found in WFDB record")
    fs = record.fs
    if fs is None:
        raise RuntimeError("Sampling frequency 'fs' not found")
    logging.info(f"Read WFDB record {wfdb_basename} (fs={fs}, shape={signals.shape})")
    return signals, fs

def select_best_lead(signals):
    if signals.ndim == 1:
        return 0
    power = np.sum(np.abs(signals), axis=0)
    best = int(np.argmax(power))
    logging.info(f"Selected lead {best} based on max power")
    return best

def custom_clean_ecg_wideband(raw, fs):
    low, high = 0.05, min(150, fs/2 - 0.1)
    if low >= high:
        return raw
    bp = nk.signal_filter(raw, sampling_rate=fs,
                          lowcut=low, highcut=high,
                          method="butterworth", order=3)
    return nk.signal_filter(bp, sampling_rate=fs,
                            method="powerline", powerline=60)

def custom_clean_ecg_for_rpeaks(raw, fs):
    low, high = 0.5, min(35, fs/2 - 0.1)
    if low >= high:
        return raw
    bp = nk.signal_filter(raw, sampling_rate=fs,
                          lowcut=low, highcut=high,
                          method="butterworth", order=4)
    return nk.signal_filter(bp, sampling_rate=fs,
                            method="powerline", powerline=60)

def simple_threshold_rpeak_detector(sig, fs):
    if np.std(sig) < 1e-4:
        return np.array([], int)
    thresh = np.percentile(sig, 95)
    mindist = int(0.3 * fs)
    peaks, _ = find_peaks(sig, height=thresh, distance=mindist)
    return peaks

def _run_delineation(sig, rpeaks, fs, method):
    try:
        _, out = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method=method)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        logging.error(f"Delineation '{method}' failed: {e}")
        return {}

def process_ecg_custom_clean(raw, fs):
    if raw is None or len(raw) == 0 or np.std(raw) < 1e-6:
        return pd.DataFrame({'ECG_Clean': raw}), np.array([], int)
    clean_wide   = custom_clean_ecg_wideband(raw, fs)
    clean_narrow = custom_clean_ecg_for_rpeaks(raw, fs)
    # R-peak detection
    rpeaks = np.array([], int)
    try:
        rd = nk.ecg_findpeaks(clean_narrow, sampling_rate=fs, method='neurokit')
        rpeaks = np.array([int(p) for p in rd.get('ECG_R_Peaks', []) if pd.notna(p)], int)
    except Exception:
        pass
    if len(rpeaks) < 3:
        alt = simple_threshold_rpeak_detector(clean_narrow, fs)
        if len(alt) >= 3:
            rpeaks = alt
    # Delineation
    waves = {}
    if len(rpeaks) >= 3:
        w_cwt  = _run_delineation(clean_wide, rpeaks, fs, 'cwt')
        w_peak = _run_delineation(clean_wide, rpeaks, fs, 'peak')
        waves['ECG_P_Onsets']    = w_peak.get('ECG_P_Onsets', [])
        waves['ECG_P_Offsets']   = w_cwt.get('ECG_P_Offsets', [])
        waves['ECG_QRS_Onsets']  = w_peak.get('ECG_QRS_Onsets', w_peak.get('ECG_Q_Peaks', []))
        waves['ECG_QRS_Offsets'] = w_peak.get('ECG_QRS_Offsets', w_peak.get('ECG_S_Peaks', []))
        waves['ECG_T_Onsets']    = w_cwt.get('ECG_T_Onsets', [])
        waves['ECG_T_Offsets']   = w_cwt.get('ECG_T_Offsets', [])
        waves['ECG_P_Peaks']     = w_peak.get('ECG_P_Peaks', [])
        waves['ECG_Q_Peaks']     = w_peak.get('ECG_Q_Peaks', [])
        waves['ECG_S_Peaks']     = w_peak.get('ECG_S_Peaks', [])
        waves['ECG_T_Peaks']     = w_peak.get('ECG_T_Peaks', [])
    df = pd.DataFrame({'ECG_Clean': clean_wide})
    for key in ['ECG_P_Onsets','ECG_P_Offsets','ECG_QRS_Onsets','ECG_QRS_Offsets',
                'ECG_T_Onsets','ECG_T_Offsets','ECG_P_Peaks','ECG_Q_Peaks',
                'ECG_S_Peaks','ECG_T_Peaks']:
        arr = np.zeros(len(clean_wide), int)
        for i in waves.get(key, []):
            if pd.notna(i) and 0 <= int(i) < len(clean_wide):
                arr[int(i)] = 1
        df[key] = arr
    return df, rpeaks

def updated_extract_waves(df):
    waves = {}
    for k in ['ECG_P_Onsets','ECG_P_Offsets','ECG_QRS_Onsets','ECG_QRS_Offsets',
              'ECG_T_Onsets','ECG_T_Offsets','ECG_P_Peaks','ECG_Q_Peaks',
              'ECG_S_Peaks','ECG_T_Peaks']:
        waves[k] = df.index[df[k] == 1].to_numpy()
    return waves

def compute_intervals(waves, fs):
    conv = 1000.0 / fs
    def median_ms(on, off, name):
        n = min(len(on), len(off))
        if n < 1:
            return None
        diffs = (off[:n] - on[:n]) * conv
        valid = diffs[(diffs > 0) &
                      (diffs >= PLAUSIBLE_RANGES[name][0]) &
                      (diffs <= PLAUSIBLE_RANGES[name][1])]
        if valid.size > 0:
            return int(round(np.nanmedian(valid)))
        return int(round(np.nanmedian(diffs))) if diffs.size > 0 else None
    return {
        'P_wave_duration_ms': median_ms(waves['ECG_P_Onsets'], waves['ECG_P_Offsets'], 'P_wave_duration_ms'),
        'PQ_interval_ms':     median_ms(waves['ECG_P_Onsets'], waves['ECG_QRS_Onsets'], 'PQ_interval_ms'),
        'QRS_duration_ms':    median_ms(waves['ECG_QRS_Onsets'], waves['ECG_QRS_Offsets'], 'QRS_duration_ms'),
        'QT_interval_ms':     median_ms(waves['ECG_QRS_Onsets'], waves['ECG_T_Offsets'], 'QT_interval_ms'),
    }

def plot_waveform_diagram(signal, fs, rpeaks, waves, title=None, filename="waveform_diagram.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    rp = int(rpeaks[0]) if len(rpeaks) > 0 else 0
    pre = min(int(0.4 * fs), rp)
    post = min(int(0.8 * fs), len(signal) - 1 - rp)
    start, end = rp - pre, rp + post + 1
    segment = signal[start:end]
    t = (np.arange(start, end) - rp) / fs
    ax.plot(t, segment, color='black', lw=1.5)
    ax.axhline(0, color='lightgray', ls='--', lw=0.75, zorder=-1)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    def gt(key):
        arr = np.array(waves.get(key, []), int)
        arr = arr[(arr >= start) & (arr < end)]
        if arr.size == 0:
            return None
        best = arr[np.argmin(np.abs(arr - rp))]
        return (best - rp) / fs
    p_on, p_off = gt('ECG_P_Onsets'), gt('ECG_P_Offsets')
    q_on, q_off = gt('ECG_QRS_Onsets'), gt('ECG_QRS_Offsets')
    t_on, t_off = gt('ECG_T_Onsets'), gt('ECG_T_Offsets')
    # bar positions
    top_bar_y = ymax + 0.05 * yrange
    top_label_y = ymax + 0.12 * yrange
    bottom_bar_y = ymin - 0.10 * yrange
    bottom_label_y = ymin - 0.15 * yrange
    qt_bar_y = bottom_bar_y + 0.03 * yrange
    qt_label_y = bottom_label_y + 0.03 * yrange
    def draw_bar(t0, t1, color, label, bar_y, lab_y):
        if t0 is None or t1 is None or t0 >= t1:
            return
        ax.hlines(bar_y, t0, t1, lw=6, color=color)
        ax.vlines([t0, t1], ymin, bar_y, ls=':', lw=1, color=color)
        ax.text((t0 + t1) / 2, lab_y, label,
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    draw_bar(p_on, q_on, 'tab:blue', 'PR Interval', bottom_bar_y, bottom_label_y)
    draw_bar(p_off, q_on, 'tab:green', 'PR Segment', top_bar_y, top_label_y)
    draw_bar(q_off, t_on, 'tab:orange', 'ST Segment', top_bar_y, top_label_y)
    draw_bar(q_on, t_off, 'tab:olive', 'QT Interval', qt_bar_y, qt_label_y)
    for x in (q_on, q_off):
        if x is not None:
            ax.axvline(x, color='gray', ls=':', lw=1)
    if q_on is not None and q_off is not None:
        ax.text((q_on + q_off) / 2, ymax + 0.08 * yrange, 'QRS Complex', ha='center', va='bottom', fontweight='bold')
    def add_point(label, t_pt, downward=False):
        if t_pt is None:
            return
        idx = np.argmin(np.abs(t - t_pt))
        yv = segment[idx]
        ax.plot(t_pt, yv, 'o', ms=8, mfc='white', mec='black', zorder=5)
        off = 0.05 * yrange
        ty = yv - off if downward else yv + off
        va = 'top' if downward else 'bottom'
        ax.text(t_pt, ty, label, ha='center', va=va, fontweight='bold')
    add_point('P', gt('ECG_P_Peaks'))
    add_point('Q', gt('ECG_Q_Peaks'), downward=True)
    add_point('R', 0, downward=False)
    add_point('S', gt('ECG_S_Peaks'), downward=True)
    add_point('T', gt('ECG_T_Peaks'))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    if title:
        ax.set_title(f"ECG Waveform Diagram ({title})", pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.grid(True, ls=':', lw=0.5, color='lightgray')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logging.info(f"Waveform diagram saved to: {filename}")

def analyze_and_plot(wfdb_basename, plot_folder, file_id):
    """
    Analyze a WFDB record and generate a plot.
    
    Args:
        wfdb_basename (str): Base path of the WFDB record
        plot_folder (str): Directory to save the plot
        file_id (str): Unique identifier for the record
        
    Returns:
        tuple: (summary dict, plot path)
    """
    # Read the WFDB record
    signals, fs = read_wfdb_record(wfdb_basename)
    
    # Select the best lead
    lead_idx = select_best_lead(signals)
    raw_signal = signals[:, lead_idx] if signals.ndim > 1 else signals
    
    # Process the ECG signal
    df, rpeaks = process_ecg_custom_clean(raw_signal, fs)
    
    # Extract wave features
    waves = updated_extract_waves(df)
    
    # Compute intervals
    intervals = compute_intervals(waves, fs)
    
    # Generate plot
    plot_path = os.path.join(plot_folder, f"{file_id}.png")
    plot_waveform_diagram(df['ECG_Clean'].values, fs, rpeaks, waves, 
                         title=f"ECG Analysis - {file_id}", 
                         filename=plot_path)
    
    # Create summary
    summary = {
        'heart_rate': int(round(60 * fs / np.median(np.diff(rpeaks)))) if len(rpeaks) > 1 else None,
        'intervals': intervals
    }
    
    return summary, plot_path
