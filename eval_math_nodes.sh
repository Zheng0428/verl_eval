#!/bin/bash

# example usage: bash eval_math.sh --run_name verl-grpo-fix-math-eval-large-reward_temp1.0_ppomicro4_Qwen2.5-14B_simplelr_math_35 --init_model Qwen2.5-14B --template qwen25-math-cot  --tp_size 1
source setup_env.sh
cd examples/simplelr_math_eval
pip uninstall latex2sympy2 -y
cd latex2sympy
pip install -e . --use-pep517
pip install Pebble
pip install sympy==1.13
pip install antlr4-python3-runtime==4.11.1
pip install timeout-decorator
pip install jieba
pip install word2number
cd ..

cd /mnt/bn/tiktok-mm-5/aiic/users/tianyu/verl_eval/examples/simplelr_math_eval
export NCCL_DEBUG=warn
# 定义评估脚本路径
set -x
add_step_0=false
temperature=0.0
max_tokens=3000
top_p=1
# benchmarks="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
benchmarks="aime24"
# output_suffix=""
output_dir="eval_results"
overwrite=false
n_sampling=1
specific_steps=""
convert_model=false  # New parameter to control model conversion
split="test"
n_test_sample=-1
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --init_model)
            INIT_MODEL_PATH="$2"
            shift 2
            ;;
        --template)
            template="$2"
            shift 2
            ;;
        --tp_size)
            tp_size="$2"
            shift 2
            ;;
        --temperature)
            temperature="$2"
            shift 2
            ;;
        --top_p)
            top_p="$2"
            shift 2
            ;;
        --max_tokens)
            max_tokens="$2"
            shift 2
            ;;
        --add_step_0)
            add_step_0="$2"
            shift 2
            ;;
        --benchmarks)
            benchmarks="$2"
            shift 2
            ;;
        --just_wandb)
            just_wandb="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --overwrite)
            overwrite="$2"
            shift 2
            ;;
        --n_sampling)
            n_sampling="$2"
            shift 2
            ;;
        --specific_steps)
            specific_steps="$2"
            shift 2
            ;;
        --convert_model)
            convert_model="$2"
            shift 2
            ;;
        --split)
            split="$2"
            shift 2
            ;;
        --n_test_sample)
            n_test_sample="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$RUN_NAME" ] || [ -z "$INIT_MODEL_PATH" ] || [ -z "$template" ] || [ -z "$tp_size" ]; then
    echo "Missing required parameters. Usage:"
    echo "--run_name <run_name> --init_model <init_model> --template <template> --tp_size <tp_size>"
    exit 1
fi


eval_script_path="sh/eval.sh"

# HDFS_HOME="/mnt/hdfs/byte_data_seed_azureb_tteng/user/huangyuzhen/SimpleRL-verl"
# 需要测试的模型
# RUN_NAME="debug_ppo_3node_long_gen_1"
base_checkpoint_path="${DEFAULT_HDFS_CHECKPOINT_PATH}/${RUN_NAME}"

# 定义初始化模型路径, SFT模型
init_model_path="${DEFAULT_HDFS_MODEL_PATH}/${INIT_MODEL_PATH}"
chmod +x sh/convert_and_evaluate_gpu_nodes.sh

TOTAL_NODES=${ARNOLD_WORKER_NUM:-8}  # Default to 1 if not set
CURRENT_NODE=${ARNOLD_ID:-0}  # Default to 0 if not set
# TOTAL_NODES=2 # Default to 1 if not set
# CURRENT_NODE=0  # Default to 0 if not set
# Add step 0 if requested - with synchronization across nodes
# if [ "${add_step_0:-false}" = true ]; then
#     done_file="$base_checkpoint_path/global_step_0/actor/huggingface/.cp_done"
    
#     if [ "$CURRENT_NODE" -eq 0 ]; then
#         # Node 0 handles the copying
#         if [ ! -f "$done_file" ]; then
#             mkdir -p "$base_checkpoint_path/global_step_0/actor/huggingface"
#             cp -r "$init_model_path"/* "$base_checkpoint_path/global_step_0/actor/huggingface/"
#             if [ $? -eq 0 ]; then
#                 touch "$done_file"
#                 echo "Copied initial model to $base_checkpoint_path/global_step_0/actor/huggingface/"
#             else
#                 echo "Failed to copy initial model"
#                 exit 1
#             fi
#         fi
#     else
#         # Other nodes wait for the .cp_done file
#         echo "Node $CURRENT_NODE waiting for step 0 files to be copied..."
#         while [ ! -f "$done_file" ]; do
#             sleep 5
#         done
#         echo "Node $CURRENT_NODE detected step 0 files are ready"
#     fi
# fi





# Before calling convert_and_evaluate_gpu.sh, get all checkpoints and distribute them
get_all_checkpoints() {
    local base_path="$1"
    local specific_steps="$2"
    local checkpoints=()
    
    # If specific steps are provided, only collect those checkpoints
    if [ -n "$specific_steps" ]; then
        IFS=',' read -r -a step_array <<< "$specific_steps"
        for step in "${step_array[@]}"; do
            step_dir="$base_path/global_step_$step"
            if [ -d "$step_dir" ]; then
                checkpoints+=("global_step_$step")
            else
                echo "Warning: Requested step $step does not exist at $step_dir"
            fi
        done
    else
        # Otherwise, collect all checkpoints
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")        
                current_step=$(echo "$step_tag" | sed 's/global_step_//')
                if (( current_step >= 80 )); then
                    checkpoints+=("$step_tag")
                fi
                checkpoints+=("$step_tag")
            fi
        done
    fi
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo ""
    else
        # Sort the checkpoints to ensure consistent ordering across nodes
        printf "%s\n" "${checkpoints[@]}" | sort -V
    fi
}
# Get all checkpoints

readarray -t all_checkpoints < <(get_all_checkpoints "$base_checkpoint_path" "$specific_steps")

total_ckpts=${#all_checkpoints[@]}

if [ $total_ckpts -eq 0 ]; then
    echo "No checkpoints found to evaluate."
    exit 0
fi

echo "Total checkpoints: $total_ckpts"
echo "Running on node $CURRENT_NODE of $TOTAL_NODES nodes"

# Distribute checkpoints across nodes
declare -a node_checkpoints
for ((i=0; i<${#all_checkpoints[@]}; i++)); do
    if [ $((i % TOTAL_NODES)) -eq $CURRENT_NODE ]; then
        node_checkpoints+=("${all_checkpoints[i]}")
    fi
done
echo "This node will evaluate ${#node_checkpoints[@]} checkpoints:"
printf '%s\n' "${node_checkpoints[@]}"
# Create a temporary file with the assigned checkpoints
tmp_ckpt_file=$(mktemp)
printf '%s\n' "${node_checkpoints[@]}" > "$tmp_ckpt_file"

if [ "$just_wandb" != "true" ]; then
    # # 调用转化和评估脚本
    bash sh/convert_and_evaluate_gpu_nodes.sh \
    "$eval_script_path" \
    "$base_checkpoint_path" \
    "$init_model_path" \
    "$template" \
    "$benchmarks" \
    "$temperature" \
    "$max_tokens" \
    "$top_p" \
    "$tp_size" \
    "$tmp_ckpt_file" \
    "$output_dir" \
    "$overwrite" \
    "$n_sampling" \
    "$convert_model" \
    "$split" \
    "$n_test_sample" 
fi

# # Log results to wandb
# export WANDB_OFFICIAL=1
# export WANDB_API_KEY=78c280e2fa597b45660678c48f3dfe054930af18
# Only run wandb logging on the first node to avoid duplicates
# if  [ "$just_wandb" = "true" ]; then
    # Log results to wandb
# export WANDB_OFFICIAL=1
# export WANDB_API_KEY=78c280e2fa597b45660678c48f3dfe054930af18
output_dir_ts="/mnt/bn/tiktok-mm-5/aiic/users/tianshun/verl/Exp/infer_result"
export PROJECT_NAME="Reinforcement-Learning"
export RUN_NAME="vllm-v1-0519-on-policy-large-off-policy-maximum-rejection-sampling_rmclipTrue_batch512_ppomini64_rolln64_maxres4096_maxvalres8192_deepscaler_train_simplelr_math_35_train_Qwen2.5-7B"
python sh/collect_results.py \
    --base_dir "$output_dir_ts/$output_dir" \
    --model_name $init_model_path \
    --wandb_project "verl_math_evaluate" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --wandb_run_name $RUN_NAME \
    --temperature $temperature \
    --benchmarks $benchmarks \
    --use_wandb 
# fi