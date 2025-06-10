import os
import pandas as pd
from tqdm import tqdm
import sys

# Add app folder for local imports
sys.path.append("./app")
from services/ecg_processing_service import ECGProcessingService


def batch_extract_features_from_ecg(
    wfdb_root_folder, output_csv, plot_folder="plots", max_new=10000
):
    """
    Process all .hea files under nested folders, extract ECG features, and continue from where it left off.
    Only processes up to `max_new` new ECGs per run.
    """
    os.makedirs(plot_folder, exist_ok=True)
    processed_ids = set()

    # Load existing CSV if it exists
    if os.path.exists(output_csv):
        processed_df = pd.read_csv(output_csv, on_bad_lines="skip")
        processed_ids = set(processed_df["file_id"].astype(str))
        print(f"🔁 Resuming — {len(processed_ids)} ECGs already processed.")
    else:
        print("🚀 Starting fresh.")

    new_count = 0

    for root, dirs, files in os.walk(wfdb_root_folder):
        for file in files:
            if file.endswith(".hea"):
                file_stem = os.path.splitext(file)[0]
                file_id = os.path.basename(root)

                if file_id in processed_ids:
                    continue  # skip already processed ECGs

                try:
                    record_basename = os.path.join(root, file_stem)

                    summary, _ = ECGProcessingService.analyze_and_plot(record_basename, plot_folder, file_id)
                    features = summary.get("physionet_features", {})

                    row = {
                        "file_id": file_id,
                        "heart_rate": summary.get("heart_rate"),
                        "rr_interval": features.get("rr_interval"),
                        "p_duration": (
                            (features.get("p_end") - features.get("p_onset"))
                            if features.get("p_end") is not None
                            and features.get("p_onset") is not None
                            else None
                        ),
                        "pq_interval": (
                            features.get("qrs_onset") - features.get("p_onset")
                            if features.get("qrs_onset") and features.get("p_onset")
                            else None
                        ),
                        "qrs_duration": (
                            features.get("qrs_end") - features.get("qrs_onset")
                            if features.get("qrs_end") and features.get("qrs_onset")
                            else None
                        ),
                        "qt_interval": (
                            features.get("t_end") - features.get("qrs_onset")
                            if features.get("t_end") and features.get("qrs_onset")
                            else None
                        ),
                        "p_axis": features.get("p_axis"),
                        "qrs_axis": features.get("qrs_axis"),
                        "t_axis": features.get("t_axis"),
                    }

                    if (
                        row["heart_rate"] is not None
                        and row["qrs_duration"] is not None
                    ):
                        df_row = pd.DataFrame([row])
                        write_header = (
                            not os.path.exists(output_csv)
                            or os.stat(output_csv).st_size == 0
                        )
                        df_row.to_csv(
                            output_csv, mode="a", index=False, header=write_header
                        )

                        new_count += 1
                        print(f"✅ Processed {file_id} (New count: {new_count})")
                    else:
                        print(
                            f"⚠️ Skipping {file_id}: missing heart rate or QRS duration"
                        )

                    if new_count >= max_new:
                        print(f"🛑 Reached {max_new} new ECGs. Stopping.")
                        return

                except Exception as e:
                    print(f"⚠️ Failed to process {file_id}: {e}")

    print(f"\n✅ Finished. Extracted {new_count} new ECG records.")
    print(f"📁 Output saved to: {output_csv}")


if __name__ == "__main__":
    input_dir = "data/external/cleaned-dataset/relevant_data/files"
    output_csv = "data/external/cleaned-dataset/final1_ecg_features_dataset.csv"
    batch_extract_features_from_ecg(input_dir, output_csv, max_new=30)
