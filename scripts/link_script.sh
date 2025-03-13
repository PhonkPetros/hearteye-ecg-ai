#!/bin/bash
# Define the external data folder (adjust if needed)
EXTERNAL_DATA_PATH="/g/Other computers/My Laptop/data"
# Define the name for the mount inside the project's data folder
LINK_NAME="external"

# Determine the project root directory (assumes this script is in a subdirectory like 'scripts')
PROJECT_ROOT=$(dirname "$(pwd)")
# Define the mount point as the existing 'data' folder in your project
MOUNT_POINT="$PROJECT_ROOT/data"

# Ensure the mount point exists
if [ ! -d "$MOUNT_POINT" ]; then
    echo "The directory '$MOUNT_POINT' does not exist. Please ensure the 'data' folder exists."
    exit 1
fi

# Define the full path for the junction inside the data folder
FULL_JUNCTION="$MOUNT_POINT/$LINK_NAME"

# Remove any existing file or folder at the target location
if [ -L "$FULL_JUNCTION" ] || [ -e "$FULL_JUNCTION" ]; then
    echo "Removing existing '$FULL_JUNCTION'..."
    rm -rf "$FULL_JUNCTION"
fi

# Convert paths to Windows format using cygpath for use with cmd
WIN_EXTERNAL_PATH=$(cygpath -w "$EXTERNAL_DATA_PATH")
WIN_MOUNT_POINT=$(cygpath -w "$MOUNT_POINT")
WIN_JUNCTION="$WIN_MOUNT_POINT\\$LINK_NAME"

# Debug output to verify the paths
echo "Project root: $PROJECT_ROOT"
echo "Mount point (data folder): $MOUNT_POINT"
echo "Junction path (Unix): $FULL_JUNCTION"
echo "External data path (Windows): $WIN_EXTERNAL_PATH"
echo "Mount point (Windows): $WIN_MOUNT_POINT"
echo "Junction path (Windows): $WIN_JUNCTION"

if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    echo "Detected Windows environment. Creating directory junction using mklink /J..."
    # Create a junction using the Windows command-line tool (mklink /J)
    cmd //c "mklink /J \"$WIN_JUNCTION\" \"$WIN_EXTERNAL_PATH\""
else
    echo "Creating symbolic link from '$EXTERNAL_DATA_PATH' to '$FULL_JUNCTION'..."
    ln -s "$EXTERNAL_DATA_PATH" "$FULL_JUNCTION"
fi

echo "Done!"
