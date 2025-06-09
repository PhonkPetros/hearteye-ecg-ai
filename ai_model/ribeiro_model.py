import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Activation,
    Dropout,
    Add,
    Flatten,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    CSVLogger,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
import math
import argparse


class ResidualUnit:
    def __init__(
        self,
        n_samples_out,
        n_filters_out,
        kernel_initializer="he_normal",
        dropout_keep_prob=0.8,
        kernel_size=17,
        preactivation=True,
        postactivation_bn=False,
        activation_function="relu",
    ):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding="same")(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        if n_filters_in != self.n_filters_out:
            y = Conv1D(
                self.n_filters_out,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            strides=downsample,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        if self.preactivation:
            x = Add()([x, y])
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


class ECGSequence(tf.keras.utils.Sequence):
    @classmethod
    def get_train_val_test(
        cls,
        train_hdf5,
        val_hdf5,
        test_hdf5,
        ecg_dset,
        labels_dset,
        batch_size=8,
        num_samples=None,
        random_seed=None,
    ):
        """
        Loads the train, validation, and test datasets from pre-split HDF5 files
        and returns corresponding ECGSequence instances for each.
        """
        # Limit the number of samples to `num_samples` while loading directly from the HDF5 file
        with h5py.File(train_hdf5, "r") as f:
            train_data = f[ecg_dset][:num_samples] if num_samples else f[ecg_dset][:]
            train_labels = (
                f[labels_dset][:num_samples] if num_samples else f[labels_dset][:]
            )

        with h5py.File(val_hdf5, "r") as f:
            val_data = f[ecg_dset][:num_samples] if num_samples else f[ecg_dset][:]
            val_labels = (
                f[labels_dset][:num_samples] if num_samples else f[labels_dset][:]
            )

        with h5py.File(test_hdf5, "r") as f:
            test_data = f[ecg_dset][:num_samples] if num_samples else f[ecg_dset][:]
            test_labels = (
                f[labels_dset][:num_samples] if num_samples else f[labels_dset][:]
            )

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        # Create the ECGSequence instances for train, validation, and test
        train_seq = cls(
            train_data,
            train_labels,
            train_hdf5,
            ecg_dset,
            labels_dset,
            batch_size=batch_size,
        )
        val_seq = cls(
            val_data, val_labels, val_hdf5, ecg_dset, labels_dset, batch_size=batch_size
        )
        test_seq = cls(
            test_data,
            test_labels,
            test_hdf5,
            ecg_dset,
            labels_dset,
            batch_size=batch_size,
        )

        return train_seq, val_seq, test_seq

    def __init__(
        self, ecg_data, labels_data, file_path, ecg_dset, labels_dset, batch_size=8
    ):
        self.file_path = file_path
        self.ecg_dset = ecg_dset
        self.labels_dset = labels_dset
        self.batch_size = batch_size
        self.num_samples = len(ecg_data)
        self.start_idx = 0
        self.end_idx = self.num_samples
        self.x = ecg_data
        self.y = labels_data

        self.f = h5py.File(file_path, "r")
        assert len(self.y.shape) == 1, "Labels should be a 1D array of integers"

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.num_samples)

        # Efficient loading of batches
        batch_x = self.f[self.ecg_dset][start:end]  # Load only the required batch
        batch_y = self.f[self.labels_dset][start:end]  # Load corresponding labels

        # Transpose ECG data
        batch_x = np.transpose(
            batch_x, (0, 2, 1)
        )  # (batch_size, 12, 5000) -> (batch_size, 5000, 12)
        return batch_x, batch_y

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()


def get_model(n_classes, input_shape=(5000, 12)):
    signal = Input(shape=input_shape, dtype=np.float32, name="signal")
    x = signal
    x = Conv1D(64, 16, padding="same", use_bias=False, kernel_initializer="he_normal")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x, y = ResidualUnit(1024, 128)([x, x])
    x, y = ResidualUnit(256, 196)([x, y])
    x, y = ResidualUnit(64, 256)([x, y])
    x, _ = ResidualUnit(16, 320)([x, y])
    x = Flatten()(x)
    diagn = Dense(n_classes, activation="softmax", kernel_initializer="he_normal")(x)
    model = Model(signal, diagn)
    return model


def main(args):
    random_seed = 42
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Load the data with train/val/test split
    train_seq, val_seq, test_seq = ECGSequence.get_train_val_test(
        args.train_hdf5,
        args.val_hdf5,
        args.test_hdf5,
        "ecg_data",
        "labels",
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        random_seed=random_seed,
    )

    # Build model
    model = get_model(3)  # Output size for 3 classes
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint("./best_model.keras", save_best_only=True),
        EarlyStopping(patience=5),
        ReduceLROnPlateau(patience=5, factor=0.1),
        TensorBoard(log_dir="./logs"),
        CSVLogger("training.log"),
    ]

    # Train the model
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=10,
        callbacks=callbacks,
    )

    # Save the final model
    model.save("./final_model.keras")

    # ---------------------- #
    # Test Set Evaluation
    # ---------------------- #

    test_preds = model.predict(test_seq, batch_size=args.batch_size)
    test_preds = np.argmax(test_preds, axis=-1)
    test_true = test_seq.y
    test_preds = test_preds.flatten()

    print("\n--- Test Set Evaluation ---")

    cm = confusion_matrix(test_true, test_preds)
    print("Confusion Matrix:\n", cm)

    accuracy = accuracy_score(test_true, test_preds)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(test_true, test_preds, average=None)
    recall = recall_score(test_true, test_preds, average=None)
    f1 = f1_score(test_true, test_preds, average=None)

    for i in range(len(precision)):
        print(f"\nClass {i}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(test_true, test_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ECG model")
    parser.add_argument(
        "train_hdf5", type=str, help="Path to the HDF5 file containing train data"
    )
    parser.add_argument(
        "val_hdf5", type=str, help="Path to the HDF5 file containing validation data"
    )
    parser.add_argument(
        "test_hdf5", type=str, help="Path to the HDF5 file containing test data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for training",
    )

    args = parser.parse_args()
    main(args)
