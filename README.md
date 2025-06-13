# HeartEye ECG AI Backend
A Flask-based backend system that processes ECG data and uses machine learning models to classify ECG signals as **normal** or **abnormal**. Built for the Data and AI Minor at Inholland Haarlem.

## Team members
Ador Negash
Maike Pierick
Petors Simonyan

## Features
- Upload ECG data (.zip) via API
- Runs feature extraction and model inference (XGBoost, RandomForest)
- JWT-based user registration and authentication
- API health check endpoint
- Containerized with Docker for easy deployment
- Azure-ready for cloud deployment (already deployed, can be redeployed using Redeployment steps below)

## Installation
### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd project
```

### 2. Set Up Environment
If you do not have access to the .env file, create a `.env` file with the following variables:
```env
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

### Prerequisites

Before you start, ensure the following:

- You have access to the Azure subscription, resource group, and Azure Container Registry (ACR).
- Azure CLI is installed and configured.
- Docker is installed and running on your local machine.
- You have the `.env` file with all necessary environment variables.
- You have permission to create/delete container instances in the specified resource group.
- You have docker deamon running.

### Redeployment steps
#### 1. Log in to Azure
Open your **bash terminal** and log in to Azure CLI:
````bash
az login
````
This will open a browser window for authentication. Complete the login process.

You will be prompted to select a subscription. Please select: Keuzeonderwijs jaar 4 

#### 2. Run the deployment script
**Note:** Make sure you are in the root directory of the repository where the `deploy.sh` script and `.env` file are located before running the commands.

Make the script executable:
````bash
chmod +x deploy.sh
````

Then run the script:
````bash
./deploy.sh
````