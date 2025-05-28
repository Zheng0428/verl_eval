#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

DEFAULT_HDFS_MODEL_PATH=/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl/base_models
MODEL=$1
INPUT_FILES=$2  # Now accepts a space-separated list of input files

# Generation parameters
TEMPERATURE=$3
TOP_P=$4
MAX_TOKENS=$5
NUM_SAMPLES=$6  
PROMPT_TYPE=$7
NUM_QUERY=$8

# Hardware parameters
BATCH_SIZE=1024
TENSOR_PARALLEL_SIZE=2

# Output directory
OUTPUT_DIR="outputs/generations"
mkdir -p "$OUTPUT_DIR"

# Check if a model was provided
if [ -z "$MODEL" ]; then
    echo "Error: No model specified"
    echo "Usage: $0 MODEL_PATH \"INPUT_FILE1 INPUT_FILE2 ...\""
    exit 1
fi

# Check if input files were provided
if [ -z "$INPUT_FILES" ]; then
    echo "Error: No input files specified"
    echo "Usage: $0 MODEL_PATH \"INPUT_FILE1 INPUT_FILE2 ...\""
    exit 1
fi

# Extract model name for output file naming
MODEL_NAME=$(basename "$MODEL")

# Process each input file
for INPUT_FILE in $INPUT_FILES; do
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file $INPUT_FILE does not exist, skipping..."
        continue
    fi
    
    # Generate output file name based on input file name and model name
    INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${INPUT_BASENAME}_generations.jsonl"
    
    echo "Processing file: $INPUT_FILE"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run the generation script for this input file
    python genrm_train/preli_exp/vllm_generate.py \
        --model "${DEFAULT_HDFS_MODEL_PATH}/${MODEL}" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --num_samples "$NUM_SAMPLES" \
        --prompt_type "$PROMPT_TYPE" \
        --num_query "$NUM_QUERY" \
        --batch_size "$BATCH_SIZE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --seed 42 \
        --dtype "bfloat16"
    
    echo "Generation complete for $INPUT_FILE. Results saved to $OUTPUT_FILE"
    echo "---------------------------------------------"
done

echo "All generation tasks completed."