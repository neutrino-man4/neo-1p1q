#!/bin/bash

# Script to create conda environment, install requirements, and activate

set -e  # Exit on any error

ENV_NAME="quantum-cpu"
PYTHON_VERSION="3.10"
REQUIREMENTS_FILE="requirements.txt"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Check if requirements.txt exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing packages from $REQUIREMENTS_FILE"
    pip install -r $REQUIREMENTS_FILE
else
    echo "Warning: $REQUIREMENTS_FILE not found in current directory"
    echo "Skipping pip install step"
fi

echo "Environment $ENV_NAME is ready and activated!"
echo "To activate this environment in the future, run: conda activate $ENV_NAME"