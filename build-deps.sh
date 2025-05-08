#!/bin/bash

# Exit immediately on error
set -e

# Get the directory of this script and switch to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Navigate to deps/hyrise
cd deps/hyrise

# Run make all
make all

# Return to script directory root
cd "$SCRIPT_DIR"

# Ensure destination directory exists
mkdir -p libs/hsql

# Determine the file extension based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    lib_file="libsqlparser.dll"
else
    lib_file="libsqlparser.so"
fi

# Full source path
src_path="deps/hyrise/$lib_file"

# Copy the file
cp "$src_path" libs/hsql/
echo "Copied $lib_file to libs/hsql/"
