import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np

# === Load files ===
features_df = pd.read_csv("data/external/ecg_features_with_header.csv")
metadata_df = pd.read_csv("data/external/cleaned-dataset/labeled_and_cleaned_metadata.csv")

# === Clean and merge study_id ===
features_df["study_id"] = features_df["study_id"].astype(str).str.extract(r'(\d+)$')[0]
metadata_df["study_id"] = metadata_df["study_id"].astype(str)
valid_labels = metadata_df[metadata_df["label"].notna()]
merged_df = pd.merge(features_df, valid_labels, on="study_id")

# === Encode labels ===
label_encoder = LabelEncoder()
merged_df["label_encoded"] = label_encoder.fit_transform(merged_df["label"])

# === Select numeric features ===
non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'gender',
                    'label', 'label_encoded', 'original_path', 'cleaned_path']
numeric_features = [
    col for col in merged_df.columns
    if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])
]

X = merged_df[numeric_features]
X = X.loc[:, X.isna().mean() <= 0.3].fillna(X.mean())
y = merged_df["label_encoded"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Balance training data ===
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

print("\U0001F4CA Balanced class distribution in training set:")
print(pd.Series(y_train_balanced).value_counts())

# === Hyperparameter tuning for Random Forest ===
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    verbose=2,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train_balanced, y_train_balanced)
best_rf = rf_search.best_estimator_

# === Evaluate best RF model ===
rf_pred = best_rf.predict(X_test)
print("\U0001F3AF Random Forest (Hyperparameter Tuned):\n", classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# === Confusion Matrix ===
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_encoder.classes_)
disp_rf.plot(cmap="Blues", xticks_rotation=45)
plt.title("\U0001F9E0 Random Forest (Tuned) - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()
