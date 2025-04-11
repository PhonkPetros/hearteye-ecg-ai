


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load cleaned ECG features and metadata

features_df = pd.read_csv("data/external/cleaned_ecg/ecg_features1.csv")


# List of important ECG features
ecg_feature_cols = [
    'QRS_Duration_Mean', 'QRS_Duration_STD',
    'QT_Interval_Mean', 'QT_Interval_STD',
    'PQ_Interval_Mean', 'PQ_Interval_STD',
    'P_Duration_Mean', 'P_Duration_STD'
]

# Ensure these columns are numeric
for col in ecg_feature_cols:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

# =======================
# 2. Load Metadata and Merge with Features
# =======================

metadata_df = pd.read_csv("data/external/cleaned-dataset/labeled_and_cleaned_metadata.csv")

# Clean & merge
features_df["study_id"] = features_df["study_id"].astype(str).str.extract(r'(\d+)$')[0]
metadata_df["study_id"] = metadata_df["study_id"].astype(str)
merged_df = pd.merge(features_df, metadata_df[metadata_df["label"].notna()], on="study_id")

# Encode labels
label_encoder = LabelEncoder()
merged_df["label_encoded"] = label_encoder.fit_transform(merged_df["label"])

# Select numeric features
non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'gender',
                    'label', 'label_encoded', 'original_path', 'cleaned_path']
numeric_features = [col for col in merged_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])]
X = merged_df[numeric_features].fillna(merged_df[numeric_features].mean())
y = merged_df["label_encoded"]

# Apply 3D t-SNE
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000)
X_tsne_3d = tsne_3d.fit_transform(X)

# Prepare for plotting
plot_df = pd.DataFrame(X_tsne_3d, columns=["Dim1", "Dim2", "Dim3"])
plot_df["label"] = label_encoder.inverse_transform(y)

# 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = {'normal': 'blue', 'abnormal': 'green', 'with arrhythmia': 'red'}

for label in plot_df["label"].unique():
    subset = plot_df[plot_df["label"] == label]
    ax.scatter(subset["Dim1"], subset["Dim2"], subset["Dim3"],
               label=label, s=20, alpha=0.6, c=colors[label])

ax.set_title("🧭 ECG Feature Distribution (3D t-SNE)")
ax.set_xlabel("t-SNE Dim 1")
ax.set_ylabel("t-SNE Dim 2")
ax.set_zlabel("t-SNE Dim 3")
ax.legend(title="Class")
plt.tight_layout()
plt.show()
