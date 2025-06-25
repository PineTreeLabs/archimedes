#!/bin/bash

# First argument is theme (light or dark)
THEME=$1
OUTPUT_BASE_DIR=$2

# Process each notebook
find source/notebooks -name "*.ipynb" -type f | while read notebook; do
  # Get the notebook path relative to source/notebooks
  rel_path=${notebook#source/notebooks/}
  
  # Extract the directory part of the relative path (empty for root files)
  rel_dir=$(dirname "$rel_path")
  
  # If we're at the root level, dirname returns "." - we need to handle this case
  if [ "$rel_dir" = "." ]; then
    output_dir="$OUTPUT_BASE_DIR"
  else
    output_dir="$OUTPUT_BASE_DIR/$rel_dir"
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p "$output_dir"
  
  echo "Converting $notebook to $output_dir"
  
  # Convert the notebook
  ARCHIMEDES_THEME="$THEME" uv run jupyter nbconvert --to markdown \
    --output-dir "$output_dir" --execute --ExecutePreprocessor.kernel_name=archimedes \
    "$notebook"
done