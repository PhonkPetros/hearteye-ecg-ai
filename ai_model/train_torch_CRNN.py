import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

# Define ECG dataset class with max_samples argument
class ECGDataset(Dataset):
    def __init__(self, path_to_hdf5, signal_dset, label_dset, max_samples=None):
        self.file = h5py.File(path_to_hdf5, 'r')

        if max_samples is not None:
            self.x = torch.tensor(self.file[signal_dset][:max_samples], dtype=torch.float32)
            self.y = torch.tensor(self.file[label_dset][:max_samples], dtype=torch.long)
            self.max_samples = max_samples
            self.file.close()
            self.file = None  # Explicitly remove reference
        else:
            self.x = self.file[signal_dset]
            self.y = self.file[label_dset]
            self.max_samples = len(self.x)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        ecg_signal = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return ecg_signal, label

    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

# Define ECG model
class ECGModel(pl.LightningModule):
    def __init__(self, model, class_weights=None):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        self.validation_losses = []
        self.test_losses = []
        self.all_preds = []
        self.all_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        
        # Apply class weights during loss calculation
        if self.class_weights is not None:
            loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        
        if self.class_weights is not None:
            val_loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        else:
            val_loss = F.cross_entropy(y_hat, y)
        
        self.validation_losses.append(val_loss)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        
        # Apply class weights during loss calculation
        if self.class_weights is not None:
            test_loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        else:
            test_loss = F.cross_entropy(y_hat, y)
        
        self.test_losses.append(test_loss)
        preds = torch.argmax(y_hat, dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        self.log("test_loss", test_loss)
        return test_loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_losses).mean()
        self.log("avg_val_loss", avg_loss)
        self.validation_losses.clear()

    def on_test_epoch_end(self):
        # Calculate average test loss
        avg_test_loss = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg_test_loss)

        # calculate all the metrics at the end of the test phase
        accuracy = accuracy_score(self.all_labels, self.all_preds)
        f1 = f1_score(self.all_labels, self.all_preds, average='weighted')
        precision = precision_score(self.all_labels, self.all_preds, average='weighted')
        recall = recall_score(self.all_labels, self.all_preds, average='weighted')

        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(self.all_labels, self.all_preds)
        print("Confusion Matrix:")
        print(conf_matrix)

        print("\nClassification Report:")
        class_report = classification_report(self.all_labels, self.all_preds, target_names=[str(i) for i in range(len(np.unique(self.all_labels)))] )
        print(class_report)

        # Log all metrics
        self.log("test_accuracy", accuracy)
        self.log("test_f1", f1)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        # Clear lists to free up memory
        self.all_preds.clear()
        self.all_labels.clear()
        self.test_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def create_dataloaders(train_file, val_file, test_file, batch_size=32, max_samples=None):
    # Load the datasets
    train_dataset = ECGDataset(train_file, 'ecg_data', 'labels', max_samples=max_samples)
    val_dataset = ECGDataset(val_file, 'ecg_data', 'labels', max_samples=max_samples)
    test_dataset = ECGDataset(test_file, 'ecg_data', 'labels', max_samples=max_samples)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Load the model
def load_model(class_names, n_leads):
    config = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=500)
    model = ECG_CRNN(classes=class_names, n_leads=n_leads, config=config)
    return model

def get_class_weights(train_file, signal_dset, label_dset):
    train_dataset = ECGDataset(train_file, signal_dset, label_dset)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.y), y=train_dataset.y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights

# Train the model
def train_model(model, train_loader, val_loader, class_weights):
    ecg_model = ECGModel(model, class_weights = class_weights)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best_model"
    )
    
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1, callbacks=[checkpoint_callback])
    trainer.fit(ecg_model, train_loader, val_loader)
    return trainer, ecg_model


def main():
    # Set the seed for reproducibility
    set_seed(42)

    # Paths to data
    train_file = "D:/Maike/train_data.h5"
    val_file = "D:/Maike/val_data.h5"
    test_file = "D:/Maike/test_data.h5"

    # Set the max_samples to a smaller number
    max_samples = None

    # Calculate class weights based on training data
    class_weights = get_class_weights(train_file, 'ecg_data', 'labels')
                                      
    # Create data loaders (training, validation, and test)
    train_loader, val_loader, test_loader = create_dataloaders(train_file, val_file, test_file, batch_size=32, max_samples=max_samples)

    # Load the model
    class_names = ["abnormal", "normal", "with arrhythmia"]
    n_leads = 12
    base_model = load_model(class_names, n_leads)

    # Train the model
    trainer, ecg_model = train_model(base_model, train_loader, val_loader, class_weights)

    # Evaluate the model on the test data
    trainer = pl.Trainer(accelerator="auto", devices=1)
    trainer.test(ecg_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
