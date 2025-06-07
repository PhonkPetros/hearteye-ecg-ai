#!/bin/bash

# Azure Container Registry Deployment Script
# This script builds and pushes the ECG AI application to Azure Container Registry

# Set environment variables
IMAGE_NAME=hearteye-ecg-ai     # Your ECG AI application name
IMAGE_TAG=latest               # Replace with your tag (e.g., v1.0.0 or latest)
ACR_NAME=model13registry       # Your Azure Container Registry name
FULL_IMAGE_NAME=${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}

echo "🚀 Starting deployment to Azure Container Registry..."
echo "Registry: ${ACR_NAME}.azurecr.io"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Full image name: ${FULL_IMAGE_NAME}"

# Step 1: Build the Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully"

# Step 2: Tag the image with the ACR login server
echo "🏷️  Tagging image for ACR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "❌ Docker tag failed!"
    exit 1
fi

echo "✅ Image tagged successfully"

# Step 3: Log in to Azure (interactive login)
echo "🔐 Logging in to Azure..."
az login

if [ $? -ne 0 ]; then
    echo "❌ Azure login failed!"
    exit 1
fi

echo "✅ Azure login successful"

# Step 4: Log in to Azure Container Registry
echo "🔐 Logging in to Azure Container Registry..."
az acr login --name ${ACR_NAME}

if [ $? -ne 0 ]; then
    echo "❌ ACR login failed!"
    exit 1
fi

echo "✅ ACR login successful"

# Step 5: Push the image to ACR
echo "⬆️  Pushing image to ACR..."
docker push ${FULL_IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed!"
    exit 1
fi

echo "✅ Image pushed successfully to ACR!"
echo "🎉 Deployment completed!"
echo ""
echo "Your image is now available at: ${FULL_IMAGE_NAME}"
echo ""
echo "To pull this image from ACR:"
echo "docker pull ${FULL_IMAGE_NAME}" 