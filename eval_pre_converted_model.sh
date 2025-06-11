set -ex
cd /mnt/bn/tiktok-mm-5/aiic/users/tianyu/verl_eval/
source setup_env.sh
cd examples/simplelr_math_eval


GPU_SET_1="0,1,2,3"
GPU_SET_2="4,5,6,7" 

PROMPT_TYPE="qwen-boxed"
OUTPUT_BASE_DIR="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted"
temperature="1.0"
max_tokens="4096"
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

IFS=',' read -ra BENCHMARK_ARRAY <<< "$benchmarks"
REGULAR_BENCHMARKS=()
SPECIAL_BENCHMARKS=()

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
    if [[ "$benchmark" == "aime24" || "$benchmark" == "amc23" ]]; then
        SPECIAL_BENCHMARKS+=("$benchmark")
    else
        REGULAR_BENCHMARKS+=("$benchmark")
    fi
done

if [ "$temperature" = "0.0" ] || [ "$temperature" = "0" ]; then
    REGULAR_BENCHMARKS=("${REGULAR_BENCHMARKS[@]}" "${SPECIAL_BENCHMARKS[@]}")
    SPECIAL_BENCHMARKS=()
fi


GPU_ASSIGNMENT_COUNTER=0

# 在这改
for i in $(seq 160 20 180); do
    MODEL_NAME_OR_PATH="${MODEL_PATH_PREFIX}/global_step_${i}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/global_step_${i}"

    CURRENT_GPU_SET=""
    if (( GPU_ASSIGNMENT_COUNTER % 2 == 0 )); then
        CURRENT_GPU_SET="$GPU_SET_1"
    else
        CURRENT_GPU_SET="$GPU_SET_2"
    fi
    GPU_ASSIGNMENT_COUNTER=$((GPU_ASSIGNMENT_COUNTER + 1))

    echo "--- Preparing to evaluate model: ${MODEL_NAME_OR_PATH} using GPUs: ${CURRENT_GPU_SET} ---"
    echo "--- Output directory: ${OUTPUT_DIR} ---"

    EXPECTED_RESULTS_FILE="test_qwen-boxed_${N_TEST_SAMPLE}_seed0_t${temperature}_p${top_p}_s0_e-1.jsonl"
    FULL_RESULTS_PATH="${OUTPUT_DIR}/${EXPECTED_RESULTS_FILE}"

    if [ -f "${FULL_RESULTS_PATH}" ]; then
        echo "--> Detected existing results at ${FULL_RESULTS_PATH}. Skipping evaluation for ${MODEL_NAME_OR_PATH}."
        continue
    fi

    if [ ${#SPECIAL_BENCHMARKS[@]} -gt 0 ]; then
        SPECIAL_BENCHMARKS_STR=$(IFS=,; echo "${SPECIAL_BENCHMARKS[*]}")
        TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES="${CURRENT_GPU_SET}" \
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