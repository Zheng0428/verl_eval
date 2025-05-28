#!/bin/bash


source setup_env.sh

cd examples/simplelr_math_eval
pip uninstall latex2sympy2 -y
cd latex2sympy
pip install -e . --use-pep517
pip install Pebble
pip install sympy==1.12
pip install antlr4-python3-runtime==4.11.1
pip install timeout-decorator
pip install jieba
pip install matplotlib
cd ..
# export CUDA_VISIBLE_DEVICES=1

export NCCL_DEBUG=warn

set -e

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 MODEL_PATH [PROMPT_TYPE] [OUTPUT_DIR] [TEMPERATURE] [MAX_TOKENS] [TOP_P] [BENCHMARKS] [OVERWRITE]"
    echo "Example: $0 /path/to/model default ./results 0.0 1024 0.95"
    exit 1
fi

# Required argument
MODEL_PATH=$1

# Optional arguments with defaults
PROMPT_TYPE=${2:-"default"}
OUTPUT_DIR=${3:-"./math_eval_results"}
TEMPERATURE=${4:-"0.0"}
MAX_TOKENS=${5:-"1024"}
TOP_P=${6:-"0.95"}
BENCHMARKS=${7:-"gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"}
OVERWRITE=${8:-"false"}
N_SAMPLING=${9:-1}
SPLIT=${10:-"test"}
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the evaluation script
echo "Evaluating model: $MODEL_PATH"
echo "Using prompt type: $PROMPT_TYPE"
echo "Saving results to: $OUTPUT_DIR"
echo "Parameters: temp=$TEMPERATURE, max_tokens=$MAX_TOKENS, top_p=$TOP_P"
echo "Benchmarks: $BENCHMARKS"
echo "Overwrite: $OVERWRITE"

# cd examples/simplelr_math_eval
bash sh/eval.sh \
    "$PROMPT_TYPE" \
    "$MODEL_PATH" \
    "$OUTPUT_DIR" \
    "$TEMPERATURE" \
    "$MAX_TOKENS" \
    "$TOP_P" \
    "$BENCHMARKS" \
    "$OVERWRITE" \
    "$N_SAMPLING" \
    "$SPLIT" 

echo "Evaluation complete. Results saved to $OUTPUT_DIR"
