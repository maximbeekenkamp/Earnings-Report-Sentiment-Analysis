#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python not found. Please install Python and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip not found. Please install pip and try again."
    exit 1
fi

# Install required packages
pip3 install tensorflow scipy pandas matplotlib sklearn

echo "Packages installed successfully."
