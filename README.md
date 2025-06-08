# HeartEye ECG AI Project

[python  main.py --comparison_only --machine_csv "C:\Users\prota\Desktop\hearteye-ecg-ai\data\external\machine_measurements.csv"] path to your dir

## Redeployment Instructions for HeartEye ECG AI Backend

This document provides step-by-step instructions for redeploying the HeartEye ECG AI Backend application using Azure Container Instances and Docker.

---

### Prerequisites

Before you start, ensure the following:

- You have access to the Azure subscription, resource group, and Azure Container Registry (ACR).
- Azure CLI is installed and configured.
- Docker is installed and running on your local machine.
- You have the `.env` file with all necessary environment variables.
- You have permission to create/delete container instances in the specified resource group.

---
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