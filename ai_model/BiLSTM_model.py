import numpy as np
import h5py
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score


class ECGDataGenerator(Sequence):
    def __init__(self, path_to_hdf5, signal_dset, label_dset, batch_size=32, max_samples=1000):
        self.file = h5py.File(path_to_hdf5, 'r')
        self.x = self.file[signal_dset]
        self.y = self.file[label_dset]

        self.max_samples = min(len(self.x), max_samples)
        self.batch_size = batch_size
        self.indices = np.arange(self.max_samples)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.max_samples / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.max_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]

        batch_x = np.transpose(batch_x, (0, 2, 1))  # (batch, 5000, 12)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __del__(self):
        self.file.close()


def get_model(num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(5000, 12)),
        LSTM(32),
        Dense(num_classes, activation='softmax')
    ])
    return model


if __name__ == "__main__":
    # File paths (change to your actual paths)
    train_file = 'D:/Maike/ecg_train.h5'
    val_file = 'D:/Maike/ecg_val.h5'
    test_file = 'D:/Maike/ecg_test.h5'

    signal_dset = 'ecg_data'
    label_dset = 'labels'

    batch_size = 32
    max_samples = 1000
    num_classes = 3

    # Create data generators
    train_generator = ECGDataGenerator(train_file, signal_dset, label_dset, batch_size, max_samples)
    val_generator = ECGDataGenerator(val_file, signal_dset, label_dset, batch_size, max_samples)
    test_generator = ECGDataGenerator(test_file, signal_dset, label_dset, batch_size, max_samples)

    # Build and compile the model
    model = get_model(num_classes)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5)
    ]

    # Train with validation
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    model.save('final_model.h5')

    # Predict and evaluate on test data
    predictions = model.predict(test_generator, verbose=1)

    true_labels = []
    for i in range(len(test_generator)):
        _, labels = test_generator[i]
        true_labels.append(labels)
    true_labels = np.concatenate(true_labels)

    predicted_labels = np.argmax(predictions, axis=1)

    acc = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {acc:.4f}")

    report = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(num_classes)])
    print("Classification Report:")
    print(report)
