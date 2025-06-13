# HeartEye ECG AI Backend
A Flask-based backend system that processes ECG data and uses machine learning models to classify ECG signals as **normal** or **abnormal**. Built for the Data and AI Minor at Inholland Haarlem.

## Team members
* Ador Negash
* Maike Pierick
* Petors Simonyan

## Features
-   Upload ECG data (.zip) via API
-   Runs feature extraction and model inference (XGBoost, RandomForest)
-   JWT-based user registration and authentication
-   API health check endpoint
-   Containerized with Docker for easy deployment
-   Azure-ready for cloud deployment (already deployed, can be redeployed using Redeployment steps below)

---
## Feature Extraction AI
This section describes the AI pipeline for extracting features from raw ECG signals. The process is designed to prepare the data for training a deep learning model to identify key ECG waveforms. This process is handled by a set of Python scripts that perform data splitting, preprocessing, and feature generation.

### Overview

The core of the feature extraction is a series of scripts that perform the following key operations:

1.  **Patient-Aware Data Split**: The dataset is first split into training, validation, and test sets, ensuring that all records from a single patient belong to only one set. This prevents data leakage and results in a more robust and generalizable model.
2.  **Data Loading**: The feature extraction script reads raw ECG records from the MIMIC-IV-ECG database, which are stored in the `WFDB` format. It processes the records specified in the pre-split CSV files (e.g., `train_records.csv`).
3.  **Signal Preprocessing**: It cleans and standardizes the raw ECG signals to make them suitable for the model.
4.  **Windowing**: Long ECG recordings are sliced into smaller, fixed-size windows (e.g., 10 seconds).
5.  **Target Mask Generation**: For each window, it creates segmentation masks that act as the ground truth labels for the model to learn from. These masks identify the P-wave, QRS-complex, and QT-interval.
6.  **Data Saving**: The processed windows (features) and masks (labels) are saved as NumPy arrays (`.npy` files), which are then used to train the segmentation model.

### Detailed Steps

Here's a breakdown of how the feature extraction and data preparation scripts work:

#### 1. Data Splitting (`perform_patient_aware_split`)

Before any features are extracted, the `perform_patient_aware_split.py` script is run. It takes a master CSV file containing all labeled ECG records and splits them into training, validation, and test sets. The split is "patient-aware," meaning all ECGs from a single patient are kept in the same set. This is crucial for training a model that can generalize to new, unseen patients.

#### 2. Configuration and Setup

The main feature extraction script (`prepare_data_for_training`) is configured with several key parameters:

* **Paths**: Defines the paths to the input CSVs (e.g., `train_records.csv`, `val_records.csv`), the base directory of the raw WFDB ECG files, and the output directory for the processed data.
* **Processing Parameters**:
    * `TARGET_SAMPLING_RATE`: All ECGs are resampled to **500 Hz** to ensure consistency.
    * `WINDOW_LENGTH_SEC`: Each ECG is divided into **10-second** windows.
    * `NUM_LEADS`: The model processes the standard **12 leads** of an ECG.

#### 3. Signal Preprocessing (`preprocess_ecg_signals`)

Each raw ECG signal undergoes a series of preprocessing steps to remove noise and standardize the data:

* **Bandpass Filter**: A Butterworth bandpass filter (e.g., 0.005 Hz to 150 Hz) is applied to remove baseline wander and high-frequency noise.
* **Notch Filter**: A notch filter (e.g., at 60 Hz) is used to eliminate powerline interference.
* **Resampling**: The signal is resampled to the `TARGET_SAMPLING_RATE` of 500 Hz.
* **Normalization**: Each of the 12 leads is individually normalized to have a mean of 0 and a standard deviation of 1. This helps the model train more effectively.

#### 4. Windowing and Target Mask Creation

* The preprocessed signal is sliced into non-overlapping windows of `WINDOW_LENGTH_SAMPLES` (5000 samples, corresponding to 10 seconds).
* For each window, the script uses pre-annotated fiducial points (like `p_onset`, `qrs_end`, etc.) from the input CSV to create three binary masks:
    * **P-wave mask**: A binary array where `1` indicates the presence of a P-wave.
    * **QRS-complex mask**: A binary array where `1` indicates the QRS complex.
    * **QT-interval mask**: A binary array where `1` indicates the QT interval.
* These masks are stacked together, creating a target array of shape `(5000, 3)` for each 10-second window.

#### 5. Data Storage and Model Training

* The processed ECG windows (the features) are collected and saved as `X_data.npy`.
* The corresponding target masks (the labels) are saved as `Y_data.npy`.
* A metadata file, `df_info.csv`, is also saved with information about each window.
* Finally, the training script (`train_model.py`) loads these `.npy` files to train a U-Net based deep learning model, which learns to automatically identify these critical ECG components from the feature data.

This pipeline ensures that raw, noisy, and variably-sampled ECG data is transformed into a clean, uniform, and labeled dataset, perfectly suited for training a sophisticated segmentation model.

---
## Installation
### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd project
```

### 2. Set Up Environment
If you do not have access to the .env file, create a `.env` file with the following variables:
```bash
JWT_SECRET_KEY
SECRET_KEY
CORS_ORIGINS
FLASK_APP
FLASK_DEBUG
DATABASE_URL
SUPABASE_URL
SUPABASE_KEY
SUPABASE_BUCKET_NAME
AZURE_REGISTRY_NAME
AZURE_RESOURCE_GROUP
AZURE_CONTAINER_NAME
AZURE_REGISTRY_USERNAME
AZURE_REGISTRY_PASSWORD
```

### 3. Install Dependencies
```bash

pip install -r requirements.txt
```

### 4. Run the App using Docker
```bash
docker build -t hearteye-backend .
docker-compose up
```

## Redeployment Instructions for HeartEye ECG AI Backend

This document provides step-by-step instructions for redeploying the HeartEye ECG AI Backend application using Azure Container Instances and Docker.
Prerequisites

Before you start, ensure the following:

    You have access to the Azure subscription, resource group, and Azure Container Registry (ACR).

    Azure CLI is installed and configured.

    Docker is installed and running on your local machine.

    You have the .env file with all necessary environment variables.

    You have permission to create/delete container instances in the specified resource group.

    You have docker deamon running.

## Redeployment steps
### 1. Log in to Azure

Open your bash terminal and log in to Azure CLI:
```bash
az login
```

This will open a browser window for authentication. Complete the login process.

You will be prompted to select a subscription. Please select: Keuzeonderwijs jaar 4

### 2. Run the deployment script

Note: Make sure you are in the root directory of the repository where the deploy.sh script and .env file are located before running the commands.

Make the script executable:
```bash
chmod +x deploy.sh
```

Then run the script:
```bash
./deploy.sh
```

## API Documentation
The full API documentation can is included as a SwaggerUI file called index.html in the docs/swagger-ui folder.
The most relevant endpoint for using just the prediction part would be the /predict endpoint that accepts Form data including, patient name, gender, age and a zip file containing either one .edf file or one .dat and .hea file.
