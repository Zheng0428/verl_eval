set -ex

# Activate environment and install dependencies
source setup_env.sh
cd examples/simplelr_math_eval

# --- Set CUDA_VISIBLE_DEVICES for all 8 GPUs ---
# This tells your program to use GPUs 0 through 7
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# --- Evaluation Parameters Definition ---
# These parameters remain the same as defined previously
# Removed 'local' keyword as these are global script variables
PROMPT_TYPE="qwen-boxed"
OUTPUT_BASE_DIR="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted"
temperature="1.0"
max_tokens="32000"
top_p="0.95"
benchmarks="aime24"
OVERWRITE="false"
N_SAMPLING="32"
SPLIT="test"
N_TEST_SAMPLE="-1"

MODEL_PATH_PREFIX="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted"

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# Split benchmarks (though fixed to aime24 here, logic remains for extensibility)
IFS=',' read -ra BENCHMARK_ARRAY <<< "$benchmarks"
REGULAR_BENCHMARKS=() # Removed 'local'
SPECIAL_BENCHMARKS=() # Removed 'local'

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
    if [[ "$benchmark" == "aime24" || "$benchmark" == "amc23" ]]; then
        SPECIAL_BENCHMARKS+=("$benchmark")
    else
        REGULAR_BENCHMARKS+=("$benchmark")
    fi
done

# If temperature is 0, combine benchmark arrays (won't execute for temperature=1.0)
if [ "$temperature" = "0.0" ] || [ "$temperature" = "0" ]; then
    REGULAR_BENCHMARKS=("${REGULAR_BENCHMARKS[@]}" "${SPECIAL_BENCHMARKS[@]}")
    SPECIAL_BENCHMARKS=()
fi

# --- Loop through each checkpoint and run the evaluation ---
# Since each experiment uses all 8 cards, these will run one after another
for i in $(seq 200 20 280); do
    # 'local' is correctly used here as 'MODEL_NAME_OR_PATH' and 'OUTPUT_DIR'
    # are scoped to this loop iteration, acting like local variables within the loop's context.
    MODEL_NAME_OR_PATH="${MODEL_PATH_PREFIX}/global_step_${i}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/global_step_${i}"

    echo "--- Preparing to evaluate model: ${MODEL_NAME_OR_PATH} using ALL 8 GPUs (0-7) ---"
    echo "--- Output directory: ${OUTPUT_DIR} ---"

    # --- Skip mechanism: Check if evaluation results already exist ---
    EXPECTED_RESULTS_FILE="test_qwen-boxed_${N_TEST_SAMPLE}_seed0_t${temperature}_p${top_p}_s0_e-1.jsonl"
    FULL_RESULTS_PATH="${OUTPUT_DIR}/${EXPECTED_RESULTS_FILE}"

    if [ -f "${FULL_RESULTS_PATH}" ]; then
        echo "--> Detected existing results at ${FULL_RESULTS_PATH}. Skipping evaluation for ${MODEL_NAME_OR_PATH}."
        continue # Skip to the next checkpoint
    fi

    # Run special benchmarks (aime24, amc23) logic
    if [ ${#SPECIAL_BENCHMARKS[@]} -gt 0 ]; then
        SPECIAL_BENCHMARKS_STR=$(IFS=,; echo "${SPECIAL_BENCHMARKS[*]}")
        TOKENIZERS_PARALLELISM=false \
        python -u math_eval.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --data_name "${SPECIAL_BENCHMARKS_STR}" \
            --output_dir "${OUTPUT_DIR}" \
            --split "${SPLIT}" \
            --prompt_type "${PROMPT_TYPE}" \
            --num_test_sample "${N_TEST_SAMPLE}" \
            --max_tokens_per_call "${max_tokens}" \
            --seed 0 \
            --temperature "${temperature}" \
            --n_sampling "${N_SAMPLING}" \
            --top_p "${top_p}" \
            --start 0 \
            --end -1 \
            --use_vllm \
            --save_outputs \
            ${OVERWRITE_FLAG}
    fi
done

echo "--- All model checkpoint evaluations completed or skipped! ---"