import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# Load and preprocess data
features_df = pd.read_csv("data/external/ecg_features_with_header.csv")
metadata_df = pd.read_csv("data/external/cleaned-dataset/labeled_and_cleaned_metadata.csv")

# Clean study_id and merge
features_df["study_id"] = features_df["study_id"].astype(str).str.extract(r'(\d+)$')[0]
metadata_df["study_id"] = metadata_df["study_id"].astype(str)
valid_labels = metadata_df[metadata_df["label"].notna()]
merged_df = pd.merge(features_df, valid_labels, on="study_id")

# Encode labels
label_encoder = LabelEncoder()
merged_df["label_encoded"] = label_encoder.fit_transform(merged_df["label"])

# Select features
non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'gender',
                    'label', 'label_encoded', 'original_path', 'cleaned_path']
numeric_features = [
    col for col in merged_df.columns
    if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])
]

X = merged_df[numeric_features]
X = X.loc[:, X.isna().mean() <= 0.3].fillna(X.mean())
y = merged_df["label_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Balance training data
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Class weights for XGBoost
classes = np.unique(y_train_balanced)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_balanced)
class_weight_dict = dict(zip(classes, weights))

# Define models
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=3,
    use_label_encoder=False,
    random_state=42,
)

rf_model = RandomForestClassifier(random_state=42)

# XGBoost hyperparameter space
param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2.0]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

xgb_search.fit(X_train_scaled, y_train_balanced)
best_xgb = xgb_search.best_estimator_

# Ensemble voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', rf_model)
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train_balanced)
y_pred = voting_clf.predict(X_test_scaled)

# Evaluation
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Show classification report and confusion matrix
print("🔥 Classification Report:")
print(pd.DataFrame(report).transpose())

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap="Oranges", xticks_rotation=45)
plt.title("⚡ XGBoost + RF Ensemble - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()
