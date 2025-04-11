import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
import optuna


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

print("Balanced class distribution in training set:")
print(pd.Series(y_train_balanced).value_counts())

# === Hyperparameter tuning for Random Forest ===

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=3, scoring='accuracy').mean()
    return score

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Retrieve best parameters and fit the final model
print("Best hyperparameters:", study.best_params)
best_rf = RandomForestClassifier(**study.best_params, random_state=42)
best_rf.fit(X_train_balanced, y_train_balanced)

# # === Evaluate best RF model ===
rf_pred = best_rf.predict(X_test)
print("Random Forest (Hyperparameter Tuned):\n", classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# === Confusion Matrix ===
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_encoder.classes_)
disp_rf.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest (Tuned) - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()
