#!/bin/bash

# Check if user is logged into Azure
if ! az account show > /dev/null 2>&1; then
  echo "You are not logged in to Azure CLI."
  echo "Please log in by running: az login"
  exit 1
fi

# Load environment variables
if [ ! -f .env ]; then
  echo ".env file not found! Please create it with the required variables."
  exit 1
fi
source .env

# Log in to Azure Container Registry
echo "Logging into Azure Container Registry..."
az acr login --name "$AZURE_REGISTRY_NAME" || { echo "ACR login failed"; exit 1; }

# Build Docker image
echo "Building Docker image..."
docker build -t "$AZURE_REGISTRY_NAME.azurecr.io/hearteye-ecg-ai-backend:latest" . || { echo "Docker build failed"; exit 1; }

# Push image to ACR
echo "Pushing Docker image to Azure Container Registry..."
docker push "$AZURE_REGISTRY_NAME.azurecr.io/hearteye-ecg-ai-backend:latest" || { echo "Docker push failed"; exit 1; }

# Deploy to Azure Container Instance
echo "Deploying container instance..."
az container create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$AZURE_CONTAINER_NAME" \
  --image "$AZURE_REGISTRY_NAME.azurecr.io/hearteye-ecg-ai-backend:latest" \
  --registry-login-server "$AZURE_REGISTRY_NAME.azurecr.io" \
  --registry-username "$AZURE_REGISTRY_USERNAME" \
  --registry-password "$AZURE_REGISTRY_PASSWORD" \
  --os-type Linux \
  --cpu 4 \
  --memory 8 \
  --location westeurope \
  --restart-policy Always \
  --dns-name-label "${AZURE_CONTAINER_NAME}-dns" \
  --ports 5000 \
  --environment-variables \
    DATABASE_URL="$DATABASE_URL" \
    SUPABASE_URL="$SUPABASE_URL" \
    SUPABASE_KEY="$SUPABASE_KEY" \
    SECRET_KEY="$SECRET_KEY" \
    JWT_SECRET_KEY="$JWT_SECRET_KEY" \
    CORS_ORIGINS="$CORS_ORIGINS" \
    FLASK_APP="$FLASK_APP" \
    FLASK_DEBUG=0 \
    SUPABASE_BUCKET_NAME="$SUPABASE_BUCKET_NAME" \
 --only-show-errors \
 --query "id" \
 --output tsv || { echo "Deployment failed"; exit 1; }
echo "Deployment complete!"
