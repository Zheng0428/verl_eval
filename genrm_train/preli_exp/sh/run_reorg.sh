#!/bin/bash

# Default values
# MODEL_NAME="Qwen2-Math-7B-Instruct"
# DATASET="deepscaler"
# FILE_PREFIX="size_200_val_qwen25-math-cot_-1_seed0_t1.0_p0.95_s0_e-1"


# MODEL_NAME="Qwen2.5-32B-Instruct"
# DATASET="skywork-or1"
# FILE_PREFIX="size_200_val_qwen-boxed_-1_seed0_t1.0_p0.95_s0_e-1"


# MODEL_NAME="DeepSeek-R1-Distill-Qwen-7B"
# DATASET="rollout_skywork"
# FILE_PREFIX="size_200_val_deepseek-r1_-1_seed0_t1.0_s0_e-1"


# MODEL_NAME="DeepSeek-R1-Distill-Qwen-32B-new"
# DATASET="rollout_deepscaler"
# FILE_PREFIX="size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1"



MODEL_NAME="Qwen2.5-Math-7B-Instruct-qwen-boxed"
DATASET="rollout_deepscaler"
FILE_PREFIX="size_1000_val_qwen-boxed_-1_seed0_t1.0_s0_e-1"

BASE_DIR="/Users/bytedance/Desktop/verl/genrm_train/preli_exp/gpt_labeled_data_new"
GROUP_BY="question"
OUTPUT_DIR=/Users/bytedance/Desktop/verl/genrm_train/preli_exp/reorg_gpt_labeled_data_new


# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --file-prefix)
      FILE_PREFIX="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --group-by)
      GROUP_BY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Construct paths
LABELED_DATA_PATH="${BASE_DIR}/${MODEL_NAME}/${DATASET}/${FILE_PREFIX}_generate_response_-1.jsonl"
ORIGINAL_DATA_PATH="${BASE_DIR}/${MODEL_NAME}/${DATASET}/${FILE_PREFIX}.jsonl"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${DATASET}/${FILE_PREFIX}_generate_response_-1_reorg.jsonl"
mkdir -p "${OUTPUT_DIR}/${MODEL_NAME}/${DATASET}"
# Print information
echo "Running reorganization with:"
echo "  Labeled data: ${LABELED_DATA_PATH}"
echo "  Original data: ${ORIGINAL_DATA_PATH}"
echo "  Output path: ${OUTPUT_PATH}"
echo "  Group by: ${GROUP_BY}"

# Run the Python script
python genrm_train/preli_exp/reorg_gpt_labeled.py \
  --labeled_data_path "${LABELED_DATA_PATH}" \
  --original_data_path "${ORIGINAL_DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --group_by "${GROUP_BY}"

echo "Done!"