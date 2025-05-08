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
make install

# Return to script directory root
cd "$SCRIPT_DIR"
