import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.utils.class_weight import compute_class_weight



# === Load files ===
features_df = pd.read_csv("data/external/ecg_features_with_header.csv")
metadata_df = pd.read_csv("data/external/cleaned-dataset/labeled_and_cleaned_metadata.csv")

# === Clean study_id columns and filter for labeled rows
# ✅ Strip the 's' prefix to match label file
features_df["study_id"] = features_df["study_id"].astype(str).str.extract(r'(\d+)$')[0]
metadata_df["study_id"] = metadata_df["study_id"].astype(str)
valid_labels = metadata_df[metadata_df["label"].notna()]
merged_df = pd.merge(features_df, valid_labels, on="study_id")

# === Encode labels ===
label_encoder = LabelEncoder()
merged_df["label_encoded"] = label_encoder.fit_transform(merged_df["label"])

# === Select feature columns ===
non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'gender',
                    'label', 'label_encoded', 'original_path', 'cleaned_path']
numeric_features = [
    col for col in merged_df.columns
    if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])
]


non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'gender',
                    'label', 'label_encoded', 'original_path', 'cleaned_path']

numeric_features = [
    col for col in merged_df.columns
    if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])
]


X = merged_df[numeric_features]
X = X.loc[:, X.isna().mean() <= 0.3]  # Drop columns with too many NaNs
X = X.fillna(X.mean())
y = merged_df["label_encoded"]


# === Stratified Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# === Balance Training Set ===
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create the base XGBoost classifier
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=3,
    random_state=42,
    use_label_encoder=False
)

# Set up GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param_dist = {
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=80,  # explore 80 combinations
    scoring='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42
)

xgb_search.fit(X_train_balanced, y_train_balanced)
best_xgb = xgb_search.best_estimator_


# Predict and evaluate
xgb_pred = best_xgb.predict(X_test)
print("🔥 XGBoost (Tuned):\n", classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, xgb_pred)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=label_encoder.classes_)
disp_xgb.plot(cmap="Oranges", xticks_rotation=45)
plt.title("⚡ XGBoost - Tuned Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()