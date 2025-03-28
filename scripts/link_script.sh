#!/bin/bash

EXTERNAL_DATA_PATH="/g/Other computers/My Laptop/data" #Set the correct path here

if [ -z "${EXTERNAL_DATA_PATH}" ]; then
    echo "Error: EXTERNAL_DATA_PATH is not set"
    exit 1
fi

if [ ! -d "${EXTERNAL_DATA_PATH}" ]; then
    echo "Error: Directory ${EXTERNAL_DATA_PATH} does not exist"
    exit 1
fi

LINK_NAME="external"
PROJECT_ROOT=$(dirname "$(pwd)")
MOUNT_POINT="$PROJECT_ROOT/data"

if [ ! -d "$MOUNT_POINT" ]; then
    echo "The directory '$MOUNT_POINT' does not exist. Please ensure the 'data' folder exists."
    exit 1
fi

FULL_JUNCTION="$MOUNT_POINT/$LINK_NAME"

if [ -L "$FULL_JUNCTION" ] || [ -e "$FULL_JUNCTION" ]; then
    echo "Removing existing '$FULL_JUNCTION'..."
    rm -rf "$FULL_JUNCTION"
fi

WIN_EXTERNAL_PATH=$(cygpath -w "$EXTERNAL_DATA_PATH")
WIN_MOUNT_POINT=$(cygpath -w "$MOUNT_POINT")
WIN_JUNCTION="$WIN_MOUNT_POINT\\$LINK_NAME"

echo "Project root: $PROJECT_ROOT"
echo "Mount point (data folder): $MOUNT_POINT"
echo "Junction path (Unix): $FULL_JUNCTION"
echo "External data path (Windows): $WIN_EXTERNAL_PATH"
echo "Mount point (Windows): $WIN_MOUNT_POINT"
echo "Junction path (Windows): $WIN_JUNCTION"

if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    echo "Detected Windows environment. Creating directory junction using mklink /J..."
    # Create a junction using the Windows command-line tool (mklink /J)
    # Use PowerShell to handle the paths correctly
    powershell.exe -Command "cmd /c mklink /J '$WIN_JUNCTION' '$WIN_EXTERNAL_PATH'"
else
    echo "Creating symbolic link from '$EXTERNAL_DATA_PATH' to '$FULL_JUNCTION'..."
    ln -s "$EXTERNAL_DATA_PATH" "$FULL_JUNCTION"
fi

echo "Done!"