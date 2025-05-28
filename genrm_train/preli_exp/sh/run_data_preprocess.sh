#!/bin/bash

# Set variables for clarity
PYTHON_SCRIPT="genrm_train/preli_exp/data_preprocess.py"
INPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/raw_data/math_train.jsonl"
OUTPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/process_for_rollout/math_train.jsonl"
TRAIN_SIZE=5000
VAL_SIZE=1000

# INPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/raw_data/orz_math_57k_collected.json"
# OUTPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/process_for_rollout/orz_math_57k_collected.jsonl"


# INPUT_PATH="agentica-org/DeepScaleR-Preview-Dataset"
# OUTPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/process_for_rollout/DeepScaleR-Preview-Dataset.jsonl"


# INPUT_PATH="Skywork/Skywork-OR1-RL-Data"
# OUTPUT_PATH="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/process_for_rollout/Skywork-OR1-RL-Data.jsonl"

# Display what we're about to do
echo "Starting data preprocessing..."
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"

# Run the Python script
python $PYTHON_SCRIPT --data_path "$INPUT_PATH" --save_path "$OUTPUT_PATH" --train_size "$TRAIN_SIZE" --val_size "$VAL_SIZE"

# Check if the command executed successfully
if [ $? -eq 0 ]; then
    echo "Data preprocessing completed successfully!"
    echo "Processed data saved to: $OUTPUT_PATH"
else
    echo "Error: Data preprocessing failed."
fi