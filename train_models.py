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

# Confirm new class distribution
print("📊 Balanced class distribution in training set:")
print(pd.Series(y_train_balanced).value_counts())

# === Model 1: Random Forest ===
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
rf_pred = rf.predict(X_test)
print("🎯 Random Forest (Balanced):\n", classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# === Confusion Matrix: Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_encoder.classes_)
disp_rf.plot(cmap="Blues", xticks_rotation=45)
plt.title("🧠 Random Forest - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
#plt.show()



# === Model 2: XGBoost ===
xgb_model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42)
xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_pred = xgb_model.predict(X_test)
print("🔥 XGBoost (Balanced):\n", classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# === Confusion Matrix: XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=label_encoder.classes_)
disp_xgb.plot(cmap="Oranges", xticks_rotation=45)
plt.title("⚡ XGBoost - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
#plt.show()



from sklearn.naive_bayes import GaussianNB

# === Model 3: Naive Bayes ===
bayes_model = GaussianNB()
bayes_model.fit(X_train_balanced, y_train_balanced)
bayes_pred = bayes_model.predict(X_test)
print("🧪 Naive Bayes (Balanced):\n", classification_report(y_test, bayes_pred, target_names=label_encoder.classes_))

# === Confusion Matrix: Naive Bayes ===
cm_bayes = confusion_matrix(y_test, bayes_pred)
disp_bayes = ConfusionMatrixDisplay(confusion_matrix=cm_bayes, display_labels=label_encoder.classes_)
disp_bayes.plot(cmap="Purples", xticks_rotation=45)
plt.title("🧠 Naive Bayes - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
#plt.show()





# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# === Define Neural Net ===
class ECGNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ECGNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # increased dropout

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = ECGNet(input_size=X_train_tensor.shape[1], num_classes=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# === Training ===
criterion = nn.CrossEntropyLoss(weight=weights_tensor)


for epoch in range(200):  # longer training
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()  # Step the LR scheduler
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred = []
    for xb, _ in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)

# === Results ===
print("\n📊 Classification Report (Neural Net):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# Optional: Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Purples')
plt.title("🧠 Neural Net - Confusion Matrix")
plt.tight_layout()
plt.show()







print("🧾 Total rows in features file:", len(features_df))
print("🧾 Total rows in labeled file:", len(metadata_df))
print("📛 Total rows after merging with labels:", len(merged_df))
print("✅ Rows used for training:", len(X_train))
print("✅ Rows used for testing:", len(X_test))
print("🔍 Label distribution after merge:\n", merged_df['label'].value_counts())

