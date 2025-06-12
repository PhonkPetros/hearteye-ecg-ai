import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split # For splitting a single dataset if needed

# --- Configuration ---
# Paths to your PROCESSED data (output from the previous pipeline script)
# IMPORTANT: Ensure these point to the .npy files generated for your TRAIN, VALIDATION, and TEST sets.
# Example: If you saved processed training data in 'processed_ecg_data_train'
PROCESSED_TRAIN_DIR = r"E:\AiModel\processed_ecg_data_full" 
PROCESSED_VAL_DIR = r"E:\AiModel\processed_ecg_data_val"
PROCESSED_TEST_DIR = r"E:\AiModel\processed_ecg_data_test" # Optional for final evaluation

X_TRAIN_PATH = os.path.join(PROCESSED_TRAIN_DIR, "X_data.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_TRAIN_DIR, "Y_data.npy")
DF_INFO_TRAIN_PATH = os.path.join(PROCESSED_TRAIN_DIR, "df_info.csv") # Optional, not used directly in training

X_VAL_PATH = os.path.join(PROCESSED_VAL_DIR, "X_data.npy")
Y_VAL_PATH = os.path.join(PROCESSED_VAL_DIR, "Y_data.npy")

X_TEST_PATH = os.path.join(PROCESSED_TEST_DIR, "X_data.npy") # For final testing
Y_TEST_PATH = os.path.join(PROCESSED_TEST_DIR, "Y_data.npy") # For final testing

# Model and Training Parameters
# These should match the output of your data preparation script
WINDOW_LENGTH_SAMPLES = 5000 # From previous script (e.g., 10s * 500Hz)
NUM_LEADS = 12
NUM_MASKS = 3 # P-wave, QRS-complex, QT-interval

BATCH_SIZE = 16 # Adjust based on your GPU memory
EPOCHS = 20       # Reduced from 50 to prevent overfitting
LEARNING_RATE = 1e-4

MODEL_OUTPUT_DIR = r"E:\AiModel\trained_models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "ecg_segmentation_unet_best.keras")


# --- 1. Define 1D U-Net Model ---
def build_unet_1d(input_shape, num_masks):
    """
    Builds a simplified 1D U-Net model for ECG segmentation.
    input_shape: (WINDOW_LENGTH_SAMPLES, NUM_LEADS)
    num_masks: Number of output segmentation masks (e.g., 3 for P, QRS, QT)
    """
    inputs = Input(input_shape)

    # Encoder Path
    c1 = Conv1D(32, 9, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv1D(32, 9, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling1D(pool_size=2)(c1)

    c2 = Conv1D(64, 7, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv1D(64, 7, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling1D(pool_size=2)(c2)

    c3 = Conv1D(128, 5, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv1D(128, 5, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling1D(pool_size=2)(c3)

    # Bottleneck
    c_bottle = Conv1D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c_bottle = Dropout(0.2)(c_bottle)
    c_bottle = Conv1D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c_bottle)

    # Decoder Path
    u3 = UpSampling1D(size=2)(c_bottle)
    u3 = concatenate([u3, c3]) # Skip connection
    c6 = Conv1D(128, 5, activation='relu', kernel_initializer='he_normal', padding='same')(u3)
    c6 = Dropout(0.2)(c6)
    c6 = Conv1D(128, 5, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u2 = UpSampling1D(size=2)(c6)
    u2 = concatenate([u2, c2]) # Skip connection
    c7 = Conv1D(64, 7, activation='relu', kernel_initializer='he_normal', padding='same')(u2)
    c7 = Dropout(0.1)(c7)
    c7 = Conv1D(64, 7, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u1 = UpSampling1D(size=2)(c7)
    u1 = concatenate([u1, c1]) # Skip connection
    c8 = Conv1D(32, 9, activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    c8 = Dropout(0.1)(c8)
    c8 = Conv1D(32, 9, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    outputs = Conv1D(num_masks, 1, activation='sigmoid')(c8) # Sigmoid for multi-label binary classification

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Main Training Script ---
if __name__ == "__main__":
    print("--- ECG Segmentation Model Training ---")

    # --- 2. Load Processed Data ---
    print("\nLoading training data...")
    if not (os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH)):
        print(f"ERROR: Training data (.npy files) not found in {PROCESSED_TRAIN_DIR}.")
        print("Please ensure you have run the data preparation pipeline (e.g., Untitled-1.py) for your 'train_records.csv'")
        print(f"and that the OUTPUT_DIR in that script was set to '{PROCESSED_TRAIN_DIR}'.")
        exit()
    try:
        X_train = np.load(X_TRAIN_PATH)
        Y_train = np.load(Y_TRAIN_PATH)
        
        # Fix data shape mismatch by truncating to minimum length
        min_train_samples = min(X_train.shape[0], Y_train.shape[0])
        X_train = X_train[:min_train_samples]
        Y_train = Y_train[:min_train_samples]
        
        print(f"  Training data loaded: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        exit()

    print("\nLoading validation data...")
    if not (os.path.exists(X_VAL_PATH) and os.path.exists(Y_VAL_PATH)):
        print(f"ERROR: Validation data (.npy files) not found in {PROCESSED_VAL_DIR}.")
        print("Please ensure you have run the data preparation pipeline for your 'val_records.csv'")
        print(f"and that the OUTPUT_DIR in that script was set to '{PROCESSED_VAL_DIR}'.")
        exit()
    try:
        X_val = np.load(X_VAL_PATH)
        Y_val = np.load(Y_VAL_PATH)
        
        # Fix data shape mismatch by truncating to minimum length
        min_val_samples = min(X_val.shape[0], Y_val.shape[0])
        X_val = X_val[:min_val_samples]
        Y_val = Y_val[:min_val_samples]
        
        print(f"  Validation data loaded: X_val shape {X_val.shape}, Y_val shape {Y_val.shape}")
    except Exception as e:
        print(f"ERROR loading validation data: {e}")
        exit()
        
    # --- 3. Build and Compile Model ---
    print("\nBuilding U-Net model...")
    input_shape = (WINDOW_LENGTH_SAMPLES, NUM_LEADS)
    model = build_unet_1d(input_shape, NUM_MASKS)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy']) 

    model.summary()

    # --- 4. Define Callbacks ---
    print("\nSetting up callbacks...")
    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_loss',      
        save_best_only=True,     
        mode='min',              
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,              # Reduced from 10 to 5 for more aggressive early stopping
        restore_best_weights=True, 
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,              
        patience=3,              # Reduced from 5 to 3 for faster learning rate reduction
        min_lr=1e-7,             
        verbose=1
    )
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # --- 5. Train the Model ---
    print(f"\nStarting training for {EPOCHS} epochs (with early stopping)...")
    history = None # Initialize history
    try:
        history = model.fit(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, Y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        print("\nTraining finished.")
        print(f"Best model saved to: {BEST_MODEL_PATH}")
    except Exception as e_train:
        print(f"ERROR during model training: {e_train}")
        # Fallback: Try to load the best model if checkpointing saved something before error
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Attempting to load checkpointed model from {BEST_MODEL_PATH} for evaluation despite training error.")
        else:
            print("No checkpointed model found to load after training error.")
            exit()


    # --- 6. Evaluate the Model (on Validation or Test Set) ---
    # Best model weights are already restored if early_stopping's restore_best_weights=True
    # Or, load the explicitly saved best model:
    print("\nLoading best model for evaluation (either from checkpoint or end of training)...")
    try:
        # If EarlyStopping restored best weights, current model is the best.
        # If training was interrupted but a checkpoint was saved, load it.
        if not early_stopping.best_weights and os.path.exists(BEST_MODEL_PATH): # Check if best_weights were restored AND if file exists
             model = tf.keras.models.load_model(BEST_MODEL_PATH)
             print(f"Loaded model from {BEST_MODEL_PATH}")
        elif not os.path.exists(BEST_MODEL_PATH) and not early_stopping.best_weights:
            print("ERROR: No best model file found and early stopping did not restore weights. Cannot evaluate.")
            exit()
        # else: current model in memory is the one to evaluate (either best restored or last state)

    except Exception as e_load:
        print(f"Error loading saved model from {BEST_MODEL_PATH}: {e_load}. Evaluating with current model state if training completed partially.")
        # If model object exists from a partial fit, it will be used. Otherwise, this will fail later.
        if 'model' not in locals():
            print("Model object does not exist. Cannot evaluate.")
            exit()


    print("\nEvaluating model on validation set:")
    try:
        val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=1)
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
    except Exception as e_eval_val:
        print(f"ERROR evaluating on validation set: {e_eval_val}")


    # --- Optional: Evaluate on Test Set ---
    if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
        print("\nLoading test data for final evaluation...")
        try:
            X_test = np.load(X_TEST_PATH)
            Y_test = np.load(Y_TEST_PATH)
            print(f"  Test data loaded: X_test shape {X_test.shape}, Y_test shape {Y_test.shape}")
            
            print("\nEvaluating model on test set:")
            # Ensure the best model is loaded for test set evaluation
            if os.path.exists(BEST_MODEL_PATH):
                print(f"Loading best model from: {BEST_MODEL_PATH} for test evaluation.")
                best_model_for_test = tf.keras.models.load_model(BEST_MODEL_PATH)
                test_loss, test_accuracy = best_model_for_test.evaluate(X_test, Y_test, verbose=1)
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Test Accuracy: {test_accuracy:.4f}")
            else:
                print(f"Best model file not found at {BEST_MODEL_PATH}. Cannot evaluate on test set.")

        except FileNotFoundError:
            print(f"Test data .npy files not found in {PROCESSED_TEST_DIR}. Skipping test set evaluation.")
        except Exception as e_test: 
            print(f"Error loading or evaluating on test data: {e_test}")
    else:
        print(f"\nTest data .npy files not found in {PROCESSED_TEST_DIR} (checked for {X_TEST_PATH} and {Y_TEST_PATH}). Skipping test set evaluation.")

    print("\n--- Script Finished ---")

    # --- Optional: Plot training history ---
    if history is not None: # Check if history object exists
        try:
            import matplotlib.pyplot as plt # Import here to make it optional
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            if 'loss' in history.history and 'val_loss' in history.history:
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title('Model loss')
                plt.ylabel('Loss') 
                plt.xlabel('Epoch')
                plt.legend(loc='upper right') # Adjusted legend location
            else:
                print("Loss history not found for plotting.")
            
            plt.subplot(1, 2, 2)
            if 'binary_accuracy' in history.history and 'val_binary_accuracy' in history.history:
                plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
                plt.plot(history.history['val_binary_accuracy'], label='Val Accuracy')
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(loc='upper left')
            else:
                print("Accuracy history not found for plotting.")
            
            plt.tight_layout()
            plot_path = os.path.join(MODEL_OUTPUT_DIR, "training_history.png")
            plt.savefig(plot_path)
            print(f"\nTraining history plot saved to {plot_path}")
            # plt.show() 
        except ImportError:
            print("Matplotlib not installed. Skipping history plot. Install with: pip install matplotlib")
        except Exception as e_plot:
            print(f"Error plotting training history: {e_plot}.")
