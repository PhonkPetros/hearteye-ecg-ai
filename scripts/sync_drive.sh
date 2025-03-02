#!/bin/bash
# sync_drive.sh
#
# This script mounts the Google Drive remote "gdrive" folder "data"
# onto the local "data" folder in the project root directory.
#
# Requirements:
#  - rclone configured with your service account (remote "gdrive")
#  - On Windows, WinFsp must be installed
#
# Usage:
#   ./sync_drive.sh

# Function to install rclone if needed
install_rclone_if_needed() {
    if ! command -v rclone &> /dev/null; then
        echo "rclone not found. Installing rclone..."
        
        # Create temporary directory
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR"
        
        if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "mingw"* ]]; then
            # Windows installation
            echo "Downloading rclone for Windows..."
            curl -O https://downloads.rclone.org/rclone-current-windows-amd64.zip
            unzip rclone-current-windows-amd64.zip
            cd rclone-*-windows-amd64
            
            # Copy binary to a directory in PATH
            mkdir -p "$HOME/bin"
            cp rclone.exe "$HOME/bin/"
            
            echo "Adding rclone to PATH..."
            if [[ ! "$PATH" == *"$HOME/bin"* ]]; then
                export PATH="$HOME/bin:$PATH"
                echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
            fi
            
            echo "rclone installed successfully. You may need to restart your shell."
        else
            # Linux/Unix installation
            echo "Downloading and installing rclone using the official script..."
            curl https://rclone.org/install.sh | sudo bash
        fi
        
        # Return to original directory
        cd "$OLDPWD"
        rm -rf "$TMP_DIR"
        
        # Verify installation
        if ! command -v rclone &> /dev/null; then
            echo "Error: Failed to install rclone. Please install manually from https://rclone.org/install/"
            exit 1
        fi
        
        echo "rclone installed successfully."
    fi
}

# Variables
RCLONE_REMOTE="gdrive"
REMOTE_FOLDER="data"

# Ensure rclone is installed
install_rclone_if_needed

# Navigate to project root (parent directory of scripts)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Define mount point in the project root
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "mingw"* ]]; then
    MOUNT_POINT=$(cygpath -w "${PROJECT_ROOT}/data")
else
    MOUNT_POINT="${PROJECT_ROOT}/data"
fi

echo "Mounting Google Drive (${RCLONE_REMOTE}:${REMOTE_FOLDER}) onto local folder ${MOUNT_POINT}..."

# Ensure rclone is configured with service account
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:"; then
    echo "Configuring rclone with service account..."
    rclone config create "${RCLONE_REMOTE}" drive scope=drive service_account_file="${PROJECT_ROOT}/scripts/credentials/gdrive_service_account.json"
fi

if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "mingw"* ]]; then
    echo "Detected Windows environment."
    # Ensure the mount point exists.
    mkdir -p "$(cygpath -u "$MOUNT_POINT")"
    # Launch rclone mount in a new Windows command prompt window.
    cmd //c start "" rclone mount "${RCLONE_REMOTE}:${REMOTE_FOLDER}" "$MOUNT_POINT" --drive-shared-with-me --vfs-cache-mode writes
    # Wait for the mount to initialize.
    sleep 10
    # Check using cygpath to convert mount point back to Unix style for listing.
    if [ -z "$(ls -A "$(cygpath -u "$MOUNT_POINT")" 2>/dev/null)" ]; then
        echo "Error: Mount point ${MOUNT_POINT} appears empty. The mount may have failed."
        echo "Possible troubleshooting steps:"
        echo "1. Ensure WinFsp is installed"
        echo "2. Check that the service account has access to the shared drive"
        echo "3. Run 'rclone ls ${RCLONE_REMOTE}:${REMOTE_FOLDER}' to test connectivity"
        exit 1
    else
        echo "Google Drive mount appears active at ${MOUNT_POINT}."
    fi
else
    echo "Detected Linux/Unix environment."
    mkdir -p "$MOUNT_POINT"
    # Mount using daemon mode.
    rclone mount "${RCLONE_REMOTE}:${REMOTE_FOLDER}" "$MOUNT_POINT" --drive-shared-with-me --vfs-cache-mode writes --daemon
    sleep 5
    if mount | grep -q "[[:space:]]${MOUNT_POINT}[[:space:]]"; then
        echo "Google Drive mount detected at ${MOUNT_POINT}."
    else
        echo "Error: Google Drive mount not detected. Exiting."
        exit 1
    fi
fi

echo "Mount complete. Your local 'data' folder now reflects your Google Drive 'data' folder."