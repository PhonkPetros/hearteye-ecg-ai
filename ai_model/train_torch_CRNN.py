import random
import numpy as np
import torch
import pytorch_lightning as pl
import h5py
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F
import json

# Define paths to datasets
train_file = 'C:/Users/maike/hearteye/train_data.h5'
val_file = 'C:/Users/maike/hearteye/val_data.h5'
test_file = 'C:/Users/maike/hearteye/test_data.h5'

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define ECG dataset class
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
        # Convert the numpy array to a PyTorch tensor and then use .clone()
        ecg_signal = torch.tensor(self.x[idx], dtype=torch.float32).clone().detach()
        label = torch.tensor(self.y[idx]).clone().detach().to(torch.long)

        return ecg_signal, label


    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

# Create the ECG model function
def load_model(class_names, n_leads, dropout):
    config = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=500)
    model = ECG_CRNN(classes=class_names, n_leads=n_leads, config=config, dropout=dropout)
    return model

# Create dataloaders function
def create_dataloaders(train_file, val_file, test_file, batch_size=32, max_samples=None):
    # Load datasets
    train_dataset = ECGDataset(train_file, 'ecg_data', 'labels', max_samples=max_samples)
    val_dataset = ECGDataset(val_file, 'ecg_data', 'labels', max_samples=max_samples)
    test_dataset = ECGDataset(test_file, 'ecg_data', 'labels', max_samples=max_samples)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter tuning
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)

    # Load class weights from the training data
    class_weights = get_class_weights(train_file, 'ecg_data', 'labels')
    
    # Load the model with these hyperparameters
    model = load_model(class_names=["abnormal", "normal", "with arrhythmia"], n_leads=12, dropout=dropout)
    
    # Create DataLoader
    train_loader, val_loader, _ = create_dataloaders(train_file, val_file, test_file, batch_size=batch_size, max_samples=1000)
    
    # Initialize model for PyTorch Lightning
    ecg_model = ECGModel(model, class_weights=class_weights)
    
    # Define optimizer with learning rate from Optuna trial
    optimizer = torch.optim.Adam(ecg_model.parameters(), lr=lr)

    # Trainer setup with pruning callback and early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True
    )
    
    callbacks = [early_stop_callback]
    if isinstance(trial.study.pruner, optuna.pruners.BasePruner):
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        if isinstance(pruning_callback, pl.callbacks.Callback):
            callbacks.append(pruning_callback)

    # Trainer setup with pruning callback
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks
    )

    # Train the model
    trainer.fit(ecg_model, train_loader, val_loader)

    # Return the validation loss to Optuna
    return trainer.callback_metrics["val_loss"].item()

# Function to calculate class weights
def get_class_weights(train_file, signal_dset, label_dset):
    train_dataset = ECGDataset(train_file, signal_dset, label_dset)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.y), y=train_dataset.y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights

# ECG Model with class weights
class ECGModel(pl.LightningModule):
    def __init__(self, model, class_weights=None, lr=0.001):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        self.lr = lr
        self.validation_losses = []
        self.test_losses = []
        self.all_preds = []
        self.all_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Apply class weights during loss calculation
        if self.class_weights is not None:
            loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(y_hat, y)

        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        accuracy = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.class_weights is not None:
            val_loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        else:
            val_loss = F.cross_entropy(y_hat, y)
        
         # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        val_accuracy = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        
        self.validation_losses.append(val_loss)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx):
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
        avg_test_loss = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg_test_loss)

        # Calculate all the metrics at the end of the test phase
        accuracy = accuracy_score(self.all_labels, self.all_preds)
        f1 = f1_score(self.all_labels, self.all_preds, average='weighted')
        precision = precision_score(self.all_labels, self.all_preds, average='weighted')
        recall = recall_score(self.all_labels, self.all_preds, average='weighted')

        # Log all metrics
        self.log("test_accuracy", accuracy)
        self.log("test_f1", f1)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        self.all_preds.clear()
        self.all_labels.clear()
        self.test_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Main function for running the Optuna optimization
def main():
    set_seed(42)  # Set the seed for reproducibility

    # Optuna study setup
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run 20 trials 

    # Print and log the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Save the best hyperparameters to a JSON file
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)


    # Once the best hyperparameters are found, train the final model on the full dataset
    final_model = load_model(class_names=["abnormal", "normal", "with arrhythmia"], 
                             n_leads=12, dropout=best_params['dropout'])
    class_weights = get_class_weights(train_file, 'ecg_data', 'labels')
    train_loader, val_loader, test_loader = create_dataloaders(train_file, val_file, test_file, 
                                                               batch_size=best_params['batch_size'], max_samples=None)

    # Final training using the best hyperparameters with early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True
    )
    ecg_model = ECGModel(final_model, class_weights=class_weights, lr=best_params['lr'])

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_callback])
    trainer.fit(ecg_model, train_loader, val_loader)
    trainer.test(ecg_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
