set -ex
cd /mnt/bn/tiktok-mm-5/aiic/users/tianyu/verl_eval/
source setup_env.sh
cd examples/simplelr_math_eval


GPU_SET="0,1,2,3"

PROMPT_TYPE="qwen-boxed"
OUTPUT_BASE_DIR="/mnt/hdfs/tiktok_aiic/user/tianshun/verl_rl_checkpoints/vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B_hf_converted"
temperature="0"
max_tokens="4096"
top_p="1"
benchmarks="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,amc23"
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







MODEL_NAME_OR_PATH="${MODEL_PATH_PREFIX}/global_step_280"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/global_step_280"


EXPECTED_RESULTS_FILE="test_qwen-boxed_${N_TEST_SAMPLE}_seed0_t${temperature}_p${top_p}_s0_e-1.jsonl"
FULL_RESULTS_PATH="${OUTPUT_DIR}/${EXPECTED_RESULTS_FILE}"

if [ ${#REGULAR_BENCHMARKS[@]} -gt 0 ]; then
    SPECIAL_BENCHMARKS_STR=$(IFS=,; echo "${REGULAR_BENCHMARKS[*]}")
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${GPU_SET}" \
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

echo "--- All model checkpoint evaluations completed or skipped! ---"