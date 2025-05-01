import os
import re
import logging
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

STANDARD_12_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF"] + [f"V{i}" for i in range(1, 7)]


def _detect_12_lead_indices(labels):
    lead_map = {}
    for lead in STANDARD_12_LEADS:
        pattern = re.compile(rf"^(?:ECG[_\s\-]*)?{lead}$", re.IGNORECASE)
        for i, lbl in enumerate(labels):
            if pattern.match(lbl):
                lead_map[lead] = i
                break
    return lead_map


def analyze_wfdb_and_plot_summary(wfdb_basename: str,
                                  plot_folder:    str,
                                  file_id:       str) -> dict:
    # 1) Load record
    rec = wfdb.rdrecord(wfdb_basename)
    sig = rec.p_signal.astype(float)
    fs = rec.fs
    labels = rec.sig_name

    # 2) Detect 12 leads (fallback if needed)
    lead_map = _detect_12_lead_indices(labels)
    if not lead_map:
        ecg_idxs = [i for i, l in enumerate(labels) if "ECG" in l.upper()]
        if ecg_idxs:
            lead_map = {"II": ecg_idxs[0]}
            logging.warning("No standard 12-lead labels; using first ECG channel as II")
        else:
            lead_map = {"I": 0}
            logging.warning("No ECG labels; defaulting channel 0 as I")

    os.makedirs(plot_folder, exist_ok=True)
    per_lead = {}

    # 3) Compute signal quality per lead
    quality_scores = {}
    for lead, idx in lead_map.items():
        x = sig[:, idx]
        try:
            sqi = nk.ecg_quality(x, sampling_rate=fs, method="averageQRS")
            quality_scores[lead] = np.mean(sqi)
        except Exception:
            quality_scores[lead] = 0.0
        logging.info(f"Lead {lead} SQI={quality_scores[lead]:.2f}")

    # 4) Select best lead for RR calculation
    best_rr_lead = max(quality_scores, key=quality_scores.get)
    rr_intervals = None

    # 5) Process each lead: clean, detect peaks, delineate
    for lead, idx in lead_map.items():
        x = sig[:, idx]
        if x.size < 2 * fs:
            logging.error(f"Lead {lead} too short (<2s), skipping")
            continue

        # 5a) Clean: Butterworth + 60Hz notch
        clean = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
        clean = nk.signal_filter(clean, sampling_rate=fs, method="powerline", powerline=60)

        # 5b) R-peak detection with Promac + artifact correction
        _, peaks = nk.ecg_peaks(clean, sampling_rate=fs, method="promac", correct_artifacts=True)
        rpeaks = peaks.get("ECG_R_Peaks", [])
        if len(rpeaks) < 2:
            logging.warning(f"Insufficient R-peaks on {lead}, skipping")
            continue

        # Save RR for chosen lead
        if lead == best_rr_lead:
            rr_intervals = np.diff(rpeaks) * 1000.0 / fs

        # 5c) Wave delineation via CWT
        _, waves = nk.ecg_delineate(clean, rpeaks=rpeaks, sampling_rate=fs, method="cwt")

        # 5d) Extract onsets/offsets for each beat
        p_on, p_off = waves.get("ECG_P_Onsets", []), waves.get("ECG_P_Offsets", [])
        r_on, r_off = waves.get("ECG_R_Onsets", []), waves.get("ECG_R_Offsets", [])
        t_off = waves.get("ECG_T_Offsets", [])

        # 5e) Align durations per beat
        p_durs, pr_durs, qrs_durs, qt_durs = [], [], [], []
        for i in range(len(rpeaks)):
            if i < len(p_on) and i < len(p_off) and p_off[i] > p_on[i]:
                p_durs.append((p_off[i] - p_on[i]) * 1000.0 / fs)
            if i < len(p_on) and i < len(r_on) and r_on[i] > p_on[i]:
                pr_durs.append((r_on[i] - p_on[i]) * 1000.0 / fs)
            if i < len(r_on) and i < len(r_off) and r_off[i] > r_on[i]:
                qrs_durs.append((r_off[i] - r_on[i]) * 1000.0 / fs)
            if i < len(r_on) and i < len(t_off) and t_off[i] > r_on[i]:
                qt_durs.append((t_off[i] - r_on[i]) * 1000.0 / fs)

        per_lead[lead] = {
            "p_durs":   np.array(p_durs),
            "pr_durs":  np.array(pr_durs),
            "qrs_durs": np.array(qrs_durs),
            "qt_durs":  np.array(qt_durs)
        }

        # 5f) Plot raw vs. clean + R-peaks
        t = np.arange(len(x)) / fs
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(t, x, label="Raw", color="silver")
        ax.plot(t, clean, label="Clean", color="black")
        ax.scatter(t[rpeaks], clean[rpeaks], label="R-peaks", color="red", s=8)
        ax.set_title(f"{file_id} — Lead {lead}")
        ax.set_xlim(0, len(x)/fs)
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_folder, f"{file_id}_{lead}.png"), dpi=120)
        plt.close(fig)

    # 6) Ensure RR computed
    if rr_intervals is None:
        raise RuntimeError("No RR intervals computed for analysis")

    # 7) Aggregate summary metrics
    rr_med = int(round(np.median(rr_intervals)))
    # Per-interval medians across leads
    durations = {"P": [], "PR": [], "QRS": [], "QT": []}
    for vals in per_lead.values():
        for name, key in zip(["P","PR","QRS","QT"], ["p_durs","pr_durs","qrs_durs","qt_durs"]):
            if vals[key].size:
                durations[name].append(np.nanmedian(vals[key]))
    # Compute means and IQRs
    summary = {"rr_median_ms": rr_med}
    for name in ["P","PR","QRS","QT"]:
        arr = np.array(durations[name])
        mean_val = np.nanmean(arr) if arr.size else np.nan
        iqr_val = (np.nanpercentile(arr,75) - np.nanpercentile(arr,25)) if arr.size > 1 else 0
        summary[f"{name.lower()}_duration_ms"] = int(round(mean_val)) if not np.isnan(mean_val) else None
        summary[f"{name.lower()}_dispersion_iqr_ms"] = int(round(iqr_val))

    # 8) QTc correction formulas
    qt_med = summary.get("qt_duration_ms", np.nan)
    rr_sec = rr_med / 1000.0
    hr = 60.0 / rr_sec
    summary.update({
        "qtc_bazett_ms":     int(round(qt_med / np.sqrt(rr_sec))) if not np.isnan(qt_med) else None,
        "qtc_fridericia_ms": int(round(qt_med / np.power(rr_sec, 1/3))) if not np.isnan(qt_med) else None,
        "qtc_hodges_ms":     int(round(qt_med + 1.75*(hr-60))) if not np.isnan(qt_med) else None,
        "qtc_framingham_ms": int(round(qt_med + 0.154*(1-rr_sec))) if not np.isnan(qt_med) else None
    })

    logging.info(f"{file_id} summary → {summary}")
    return summary
