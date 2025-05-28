#!/bin/bash

# Run GenRM statistics in batch mode for all models and tasks
# Simplified script that just runs the Python batch processor

# Get the directory of this script
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_ROOT="$SCRIPT_DIR/../../.."

# Set python path to include the project
# export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define the input and output directories
EVAL_DIR="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_eval_whole_set"
OUTPUT_DIR="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/results_whole_set"

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the batch statistics script
echo "Running batch statistics for all models across all tasks..."
python /Users/bytedance/Desktop/verl/genrm_train/preli_exp/genrm_stats_batch.py \
    --input_dir "$EVAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --suffix "whole_set"

echo "Batch statistics completed. Results saved to $OUTPUT_DIR" 