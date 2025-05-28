#!/bin/bash

# Base paths
BASE_INPUT_DIR="genrm_train/preli_exp/reorg_gpt_labeled_data_new"
BASE_OUTPUT_DIR="genrm_train/preli_exp/gt_ans_prompt_hack"

# Example for Qwen2.5-32B-Instruct model
# MODEL="Qwen2.5-32B-Instruct"
# DATASET="orz"
# FILENAME="size_200_val_qwen-boxed_-1_seed0_t1.0_p0.95_s0_e-1_generate_response_-1_reorg"


MODEL="DeepSeek-R1-Distill-Qwen-32B-new"
DATASET="rollout_deepscaler"
OUTPUT_DIR="genrm_$(echo ${DATASET} | cut -d'_' -f2)_hack"
FILENAME="size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg"

# Create path variables
INPUT_FILE="${BASE_INPUT_DIR}/${MODEL}/${DATASET}/${FILENAME}.jsonl"
OUTPUT_FILE_WO_Q_HACK="${BASE_OUTPUT_DIR}/${MODEL}/${OUTPUT_DIR}/val-r1_qwen_32b-wo_q_false_only_hack.jsonl"
OUTPUT_FILE_W_Q_HACK="${BASE_OUTPUT_DIR}/${MODEL}/${OUTPUT_DIR}/val-r1_qwen_32b-w_q_false_only_hack.jsonl"
TIGER_HACK_FILE="${BASE_OUTPUT_DIR}/${MODEL}/${OUTPUT_DIR}/val-r1_qwen_32b-tigerverifier_false_only_hack_v3.jsonl"
# OUTPUT_FILE_XVERIFIER_W_PRED="${BASE_OUTPUT_DIR}/${MODEL}/${DATASET}/val-r1_qwen_32b-xverifier_false_only_w_pred.jsonl"

# val-r1_qwen_32b-w_q_false_only
# Run without question
# echo "Creating prompts without questions..."
# python genrm_train/preli_exp/construct_gt_ans_prompt_hack.py \
#     --input_file "${INPUT_FILE}" \
#     --output_file "${OUTPUT_FILE_WO_Q_HACK}" \
#     --prompt_format "genrm_wo_q"

# # Run with question
# echo "Creating prompts with questions..."
# python genrm_train/preli_exp/construct_gt_ans_prompt_hack.py \
#     --input_file "${INPUT_FILE}" \
#     --output_file "${OUTPUT_FILE_W_Q_HACK}" \
#     --prompt_format "genrm_w_q"






echo "Creating prompts for tigerverifier with predicted answer..."
python genrm_train/preli_exp/construct_gt_ans_prompt_hack.py \
    --input_file "${INPUT_FILE}" \
    --output_file "${TIGER_HACK_FILE}" \
    --prompt_format "tigerverifier"


echo "Done!" 