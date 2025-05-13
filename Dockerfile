# Use slim base image
FROM python:3.9-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /hearteye-ecg-ai

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Copy the requirements.txt file from your local machine to the container
# This is done before copying the rest of the app for better layer caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir reduces the image size by not caching pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# The '.' means copy everything from the current directory
COPY . .

# Inform Docker that the container will listen on port 5000
# This is a documentation feature and doesn't actually publish the port
EXPOSE 5000

# Start the app using the Flask CLI, explicitly pointing to the app file
CMD flask --app app.py run --host=0.0.0.0 --port=5000 --debug