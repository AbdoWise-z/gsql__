#!/bin/bash

# Exit on error
set -e

LINE='export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH'

# Check if the line already exists
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
    echo "Added LD_LIBRARY_PATH to ~/.bashrc"
else
    echo "LD_LIBRARY_PATH already set in ~/.bashrc"
fi

source ~/.bashrc

# Go to the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p build

# Configure the CMake project with Release mode
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build --config Release --target gsql__

# Determine the binary name (with extension on Windows)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    binary_name="gsql__.exe"
else
    binary_name="gsql__"
fi

# Copy the built binary to the script's root folder
cp "build/$binary_name" "$SCRIPT_DIR/"
echo "Copied $binary_name to project root"