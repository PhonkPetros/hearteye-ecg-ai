import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# === Load Data ===
features_df = pd.read_csv("data/external/ecg_features_with_header.csv")
metadata_df = pd.read_csv("data/external/cleaned-dataset/final_labeled_and_cleaned_metadata.csv")

# === Clean and Merge ===
features_df["study_id"] = features_df["study_id"].astype(str).str.extract(r'(\d+)$')[0]
metadata_df["study_id"] = metadata_df["study_id"].astype(str)
valid_labels = metadata_df[metadata_df["label"].notna()]
merged_df = pd.merge(features_df, valid_labels, on="study_id")


# === Encode Labels ===
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
merged_df["label_encoded"] = label_encoder.fit_transform(merged_df["label"])

# === Features & Labels ===
non_feature_cols = ['record_id', 'patient_id', 'study_id', 'file_path', 'file_path.1', 'gender', 'label', 'label_encoded', 'original_path', 'cleaned_path']
numeric_features = [col for col in merged_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(merged_df[col])]
X = merged_df[numeric_features].fillna(merged_df[numeric_features].mean())
y = merged_df["label_encoded"]

# === Train-Test Split ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# === Define Neural Net ===
class ECGNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ECGNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # increased dropout

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
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


optimizer = optim.Adam(model.parameters(), lr=0.0005)  # slightly lower LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# === Training ===
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):  # longer training
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

# === Debug: Data coverage
print("🧾 Total ECG feature rows:", len(features_df))
print("🧾 Total labeled rows:", len(valid_labels))
print("📛 Rows after merge:", len(merged_df))
print("🔍 Label distribution:\n", merged_df["label"].value_counts())
