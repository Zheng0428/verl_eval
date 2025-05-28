#!/bin/bash

# example usage: bash eval_math.sh --run_name verl-grpo-fix-math-eval-large-reward_temp1.0_ppomicro4_Qwen2.5-14B_simplelr_math_35 --init_model Qwen2.5-14B --template qwen25-math-cot  --tp_size 1
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

# 定义评估脚本路径
set -x
add_step_0=false
temperature=0.0
max_tokens=3000
top_p=1
benchmarks="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
output_dir="eval_results"
overwrite=false
just_wandb=false
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

echo "Current configuration:"
echo "HDFS_CHECKPOINT_PATH: $HDFS_CHECKPOINT_PATH"
echo "HDFS_MODEL_PATH: $HDFS_MODEL_PATH"
echo "RUN_NAME: $RUN_NAME"
echo "INIT_MODEL_PATH: $INIT_MODEL_PATH"
echo "template: $template"
echo "WANDB_API_KEY: $WANDB_API_KEY"

# RUN_NAME="debug_ppo_3node_long_gen_1"
base_checkpoint_path="${HDFS_CHECKPOINT_PATH}/${RUN_NAME}"

# 定义初始化模型路径, SFT模型
init_model_path="${HDFS_MODEL_PATH}/${INIT_MODEL_PATH}"
chmod +x sh/convert_and_evaluate_gpu.sh

# Add step 0 if requested
if [ "${add_step_0:-false}" = true ]; then
    done_file="$base_checkpoint_path/global_step_0/actor/huggingface/.cp_done"
    if [ ! -f "$done_file" ]; then
        mkdir -p "$base_checkpoint_path/global_step_0/actor/huggingface"
        cp -r "$init_model_path"/* "$base_checkpoint_path/global_step_0/actor/huggingface/"
        if [ $? -eq 0 ]; then
            touch "$done_file"
            echo "Copied initial model to $base_checkpoint_path/global_step_0/actor/huggingface/"
        else
            echo "Failed to copy initial model"
            exit 1
        fi
    fi
fi

# 调用转化和评估脚本
if [ "$just_wandb" != "true" ]; then
    bash sh/convert_and_evaluate_gpu.sh \
        "$eval_script_path" \
        "$base_checkpoint_path" \
        "$init_model_path" \
        "$template" \
        "$benchmarks" \
        "$temperature" \
        "$max_tokens" \
        "$top_p" \
        "$tp_size" \
        "$output_dir" \
        "$overwrite"
fi

python sh/collect_results.py \
    --base_dir "$base_checkpoint_path/$output_dir" \
    --model_name $init_model_path \
    --wandb_project "verl_math_evaluate" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --wandb_run_name $RUN_NAME \
    --temperature $temperature \
    --benchmarks $benchmarks \
    --use_wandb 

