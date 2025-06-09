import pandas as pd
import numpy as np
import os
import joblib

# === Load trained model ===
model_path = "models/xgb_rr_hr_optimized_model.pkl"  # adjust path if needed
model = joblib.load(model_path)
expected_features = model.feature_names_in_

# === Load ECG feature dataset (38 new ECGs) ===
input_path = "data/external/cleaned-dataset/final_heartEye_ecg_features.csv"

df = pd.read_csv(input_path)

# === Manually add age and gender_encoded for all rows ===
df["age"] = 40
df["gender_encoded"] = 0

# === Keep only expected columns ===
missing = set(expected_features) - set(df.columns)
if missing:
    raise ValueError(f"Missing features in input data: {missing}")

input_df = df[expected_features].copy()

# === Clean and clip plausible ranges if desired ===
# Example:
clip_ranges = {
    "qrs_duration": (60, 120),
    "qt_interval": (300, 450),
    "pq_interval": (50, 200),
    "p_duration": (60, 120)
}
for col, (low, high) in clip_ranges.items():
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").clip(lower=low, upper=high)

# === Handle missing values ===
input_df = input_df.fillna(input_df.median(numeric_only=True))

# === Predict ===
y_pred = model.predict(input_df)
y_prob = model.predict_proba(input_df)[:, 1]  # probability of abnormal

# === Attach results to original rows ===
results = df.copy()
results["Prediction"] = y_pred
results["Abnormal_Prob"] = y_prob
results["Prediction_Label"] = results["Prediction"].map({0: "Normal", 1: "Abnormal"})

# === Print results ===
print("\n🔍 Detailed Predictions:\n")
for i, row in results.iterrows():
    print(f"🔹 ECG {i}: {row['Prediction_Label']} (Confidence: {row['Abnormal_Prob']:.4f})")

# === Summary ===
total = len(results)
normal = (results["Prediction"] == 0).sum()
abnormal = (results["Prediction"] == 1).sum()

print("\n✅ Batch Prediction Summary:")
print(f"   Normal (0):   {normal} ({normal/total:.2%})")
print(f"   Abnormal (1): {abnormal} ({abnormal/total:.2%})")

# === Save results ===
output_path = "data/external/cleaned-dataset/Batch_Prediction_Results_Rf.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results.to_csv(output_path, index=False)
print(f"\n💾 Results saved to: {output_path}")
