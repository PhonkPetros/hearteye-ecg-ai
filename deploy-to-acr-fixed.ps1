# Azure Container Registry Deployment Script (PowerShell)
# This script builds and pushes the ECG AI application to Azure Container Registry

# Set environment variables
$IMAGE_NAME = "hearteye-ecg-ai"
$IMAGE_TAG = "latest"
$ACR_NAME = "model13registry"
$FULL_IMAGE_NAME = "$ACR_NAME.azurecr.io/$IMAGE_NAME`:$IMAGE_TAG"

Write-Host "Starting deployment to Azure Container Registry..." -ForegroundColor Green
Write-Host "Registry: $ACR_NAME.azurecr.io" -ForegroundColor Cyan
Write-Host "Image: $IMAGE_NAME`:$IMAGE_TAG" -ForegroundColor Cyan
Write-Host "Full image name: $FULL_IMAGE_NAME" -ForegroundColor Cyan

# Step 1: Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t "$IMAGE_NAME`:$IMAGE_TAG" .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Docker image built successfully" -ForegroundColor Green

# Step 2: Tag the image with the ACR login server
Write-Host "Tagging image for ACR..." -ForegroundColor Yellow
docker tag "$IMAGE_NAME`:$IMAGE_TAG" $FULL_IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker tag failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Image tagged successfully" -ForegroundColor Green

# Step 3: Log in to Azure
Write-Host "Logging in to Azure..." -ForegroundColor Yellow
az login

if ($LASTEXITCODE -ne 0) {
    Write-Host "Azure login failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Azure login successful" -ForegroundColor Green

# Step 4: Log in to Azure Container Registry
Write-Host "Logging in to Azure Container Registry..." -ForegroundColor Yellow
az acr login --name $ACR_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "ACR login failed!" -ForegroundColor Red
    exit 1
}

Write-Host "ACR login successful" -ForegroundColor Green

# Step 5: Push the image to ACR
Write-Host "Pushing image to ACR..." -ForegroundColor Yellow
docker push $FULL_IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Image pushed successfully to ACR!" -ForegroundColor Green
Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Your image is now available at: $FULL_IMAGE_NAME" -ForegroundColor Cyan
Write-Host ""
Write-Host "To pull this image from ACR:" -ForegroundColor Yellow
Write-Host "docker pull $FULL_IMAGE_NAME" -ForegroundColor White 