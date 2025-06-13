# Prepare updated Python code string that includes heart_rate and rr_interval as features
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import joblib
import shap

# === Load dataset ===

df = pd.read_csv("data/external/cleaned-dataset/Balanced_Synthetic_Dataset.csv")

# === Encode gender and binary label ===
df["gender_encoded"] = df["gender"].map({"M": 0, "F": 1})
df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# === Drop irrelevant columns ===
columns_to_drop = [
    "file_id",
    "study_id",
    "file_path",
    "file_path.1",
    "Unnamed: 0",
    "label",
    "gender",
]
df = df.drop(columns=columns_to_drop)

# === Prepare features and label ===
selected_features = [
    "heart_rate",
    "rr_interval",  # newly added
    "p_duration",
    "pq_interval",
    "qrs_duration",
    "qt_interval",
    "p_axis",
    "qrs_axis",
    "t_axis",
    "age",
    "gender_encoded",
]
X = df[selected_features].copy()
y = df["binary_label"]

# === Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Oversample training data ===
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# === Hyperparameter search space ===
param_dist = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1],
    "min_child_weight": [1, 3],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_clf = XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42,
)

search.fit(X_train_bal, y_train_bal)
best_xgb = search.best_estimator_

# === Predict probabilities and test thresholds ===
y_prob = best_xgb.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.35, 0.4, 0.5]:
    print(f"\\n📊 Classification Report (Threshold = {threshold:.2f})")
    y_pred = (y_prob >= threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["normal", "abnormal"]
    ).plot(cmap="Oranges")
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# === SHAP Analysis ===
explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)

# === Save model ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_xgb, "models/xgb_rr_hr_optimized_model_v2.pkl")
