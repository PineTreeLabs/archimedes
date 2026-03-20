#!/usr/bin/env bash
# Download and extract CALCE SP20 battery dataset
# Data is available for research use - cite: https://calce.umd.edu/battery-data
#
# Usage: bash get_data.sh
#   Run from the same directory as battery-sysid.md

set -euo pipefail

CALCE_DIR="calce"
BASE_URL="https://web.calce.umd.edu/batteries/data"

ZIPS=(
    "SP1_25C_IC_OCV_12_2_2015.zip"
    "SP2_25C_DST.zip"
    "SP2_25C_FUDS.zip"
    "SP2_25C_US06.zip"
)

mkdir -p "$CALCE_DIR"

for zip in "${ZIPS[@]}"; do
    dest="$CALCE_DIR/$zip"
    if [ -f "$dest" ]; then
        echo "Already downloaded: $zip"
    else
        echo "Downloading $zip..."
        curl -L --fail --progress-bar -o "$dest" "$BASE_URL/$zip"
    fi
    echo "Extracting $zip..."
    unzip -q -o "$dest" -d "$CALCE_DIR"
done

echo "Processing OCV curve..."
cd "$CALCE_DIR" && uv run python process_ocv.py && cd ..

echo "Done. Data is in $CALCE_DIR/"
